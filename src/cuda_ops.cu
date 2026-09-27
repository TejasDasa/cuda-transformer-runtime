#include <cuda_runtime.h>
#include <cmath>

#include "cuda_ops.hpp"
#include "attention_shape.hpp"

#include <cstddef>

__global__ void matvec_kernel(float* output, const float* matrix, const float* input, int rows, int cols)
{
    __shared__ float partial[32];

    int i = threadIdx.x;
    int row = blockIdx.x;
    int j = i;
    float local_sum = 0.0f;

    while (j < cols) {
        local_sum += matrix[row * cols + j] * input[j];
        j += blockDim.x;
    }
    partial[i] = local_sum;

    __syncthreads();


    int stride = 16;

    while (stride > 0) {
        if (i < stride) {
            partial[i] += partial[i + stride];
        }

        __syncthreads();
        stride = stride / 2;
    }

    if (i == 0) {
        output[row] = partial[0];
    }
}



__global__ void rmsnorm_kernel(float* output, const float* input, const float* weights, int size, float epsilon)
{
    __shared__ float partial[32];

    int i = threadIdx.x;

    float local_sum = 0.0f;
    int j = i;

    while (j < size) {
        local_sum += input[j] * input[j];
        j += blockDim.x;
    }
    partial[i] = local_sum;

    __syncthreads();



    int stride = 16;

    while (stride > 0) {
        if (i < stride) {
            partial[i] += partial[i + stride];
        }

        __syncthreads();
        stride = stride / 2;
    }

    float scale = 1 / std::sqrt(partial[0] / size + epsilon);
    j = i;

    while (j < size) {
        output[j] = weights[j] * input[j] * scale;
        j += blockDim.x;
    }
}

// One thread per adjacent pair.  Thread t (flattened over all blocks)
// owns pair number t of the whole vector:
//     head = t / pairs_per_head        which head the pair lives in
//     pair = t % pairs_per_head        pair index inside that head
// so its two elements are vec[head * head_size + 2 * pair] and the one
// after it.  Every element belongs to exactly one thread, so reading both
// into registers and then writing both back is race-free even though the
// update is in place.
__global__ void rope_kernel(float* vec, int n_heads, int head_size, const float* cos_row, const float* sin_row)
{
    const int pairs_per_head = head_size / 2;
    const int total_pairs = n_heads * pairs_per_head;

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_pairs) {
        return;   // the last block is usually only partly filled
    }

    const int head = t / pairs_per_head;
    const int pair = t % pairs_per_head;
    const int base = head * head_size + 2 * pair;

    const float c = cos_row[pair];
    const float s = sin_row[pair];

    const float a = vec[base];
    const float b = vec[base + 1];

    vec[base]     = a * c - b * s;
    vec[base + 1] = a * s + b * c;
}


cudaError_t matvec_cuda(
    float* output,
    const float* matrix,
    const float* input,
    int rows,
    int cols
)
{
    matvec_kernel<<<rows, 32>>>(output, matrix, input, rows, cols);
    return cudaGetLastError();
}


cudaError_t rmsnorm_cuda(
    float* output, 
    const float* input, 
    const float* weights, 
    int size, 
    float epsilon
)
{
    rmsnorm_kernel<<<1, 32>>>(output, input, weights, size, epsilon);
    return cudaGetLastError();
}


cudaError_t rope_cuda(
    float* vec,
    int n_heads,
    int head_size,
    const float* cos_row,
    const float* sin_row
)
{
    const int total_pairs = n_heads * (head_size / 2);
    if (total_pairs <= 0) {
        return cudaSuccess;   // nothing to rotate; a 0-block launch is invalid
    }

    constexpr int block_size = 256;
    const int blocks = (total_pairs + block_size - 1) / block_size;   // round up

    rope_kernel<<<blocks, block_size>>>(vec, n_heads, head_size, cos_row, sin_row);
    return cudaGetLastError();
}


// =====================================================================
// KV cache and attention kernels.
//
// Kernel ordering for one token step (all on the default stream, which
// runs launches in issue order, so each kernel sees the previous one's
// results without extra synchronisation):
//     rope(q), rope(k)
//     kv_cache_store(k), kv_cache_store(v)
//     attention_scores  -> scores [n_heads, seq_len]
//     softmax_rows      -> probs  [n_heads, seq_len]
//     attention_output  -> output [n_heads, head_size]
// =====================================================================

// Block sizes.  The reductions below assume a power of two.
constexpr int kStoreBlock = 256;
constexpr int kScoreBlock = 128;
constexpr int kSoftmaxBlock = 256;
constexpr int kOutputBlock = 128;

// Sums `partial[0..blockDim.x)` into partial[0] with a tree reduction.
// Every thread must call this (it contains barriers).  blockDim.x must
// be a power of two.
__device__ void block_sum(float* partial)
{
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
}

// Same shape as block_sum but keeps the maximum.
__device__ void block_max(float* partial)
{
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
}

// One thread per float of the row.  Row pos begins pos * kv_dim floats
// into the cache; the multiplication is done in size_t on purpose.
__global__ void kv_cache_store_kernel(float* cache, const float* current, int kv_dim, int pos)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        cache[static_cast<std::size_t>(pos) * kv_dim + i] = current[i];
    }
}

// Grid (n_heads, active_len): block (h, t) computes one score, the dot
// product of query head h with cached key row t of its KV head.  Threads
// stride over head_size (so any head size works), then reduce.
__global__ void attention_scores_kernel(
    float* scores,
    const float* q,
    const float* k_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len
)
{
    __shared__ float partial[kScoreBlock];

    const int h = blockIdx.x;
    const int t = blockIdx.y;

    // Grouped-query mapping: consecutive query heads share a KV head.
    const int kv_head = h / (n_heads / n_kv_heads);
    const std::size_t kv_dim = static_cast<std::size_t>(n_kv_heads) * head_size;

    const float* q_head = q + static_cast<std::size_t>(h) * head_size;
    const float* k_row = k_cache + t * kv_dim + static_cast<std::size_t>(kv_head) * head_size;

    float local = 0.0f;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        local += q_head[d] * k_row[d];
    }
    partial[threadIdx.x] = local;
    __syncthreads();

    block_sum(partial);

    if (threadIdx.x == 0) {
        scores[static_cast<std::size_t>(h) * seq_len + t] =
            partial[0] / sqrtf(static_cast<float>(head_size));
    }
}

// One block per row.  Threads stride over the active_len entries, so a
// row longer than the block is handled by the loops, not by assumption.
// Three passes: (1) row maximum, (2) exponentials and their sum,
// (3) normalisation.  Shared memory is reused between the two
// reductions, hence the extra barrier after reading the maximum.
__global__ void softmax_rows_kernel(
    float* probs,
    const float* scores,
    int row_stride,
    int active_len
)
{
    __shared__ float partial[kSoftmaxBlock];

    const float* score_row = scores + static_cast<std::size_t>(blockIdx.x) * row_stride;
    float* prob_row = probs + static_cast<std::size_t>(blockIdx.x) * row_stride;

    // (1) maximum.  Threads with no entries contribute -inf, which is the
    // identity for max.
    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < active_len; t += blockDim.x) {
        local_max = fmaxf(local_max, score_row[t]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    block_max(partial);
    const float row_max = partial[0];
    __syncthreads();   // everyone has read partial[0] before it is reused

    // (2) exponentials.  score - max <= 0, so expf never overflows.
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < active_len; t += blockDim.x) {
        const float e = expf(score_row[t] - row_max);
        prob_row[t] = e;
        local_sum += e;
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();
    block_sum(partial);
    const float row_sum = partial[0];

    // (3) normalise.  Each thread touches only the entries it wrote.
    for (int t = threadIdx.x; t < active_len; t += blockDim.x) {
        prob_row[t] /= row_sum;
    }
}

// One block per query head; thread owns output[h, d] for its d values
// and walks the visible cache rows sequentially.  No two threads write
// the same output element, so no reduction or atomics are needed.
__global__ void attention_output_kernel(
    float* output,
    const float* probs,
    const float* v_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int active_len
)
{
    const int h = blockIdx.x;
    const int kv_head = h / (n_heads / n_kv_heads);
    const std::size_t kv_dim = static_cast<std::size_t>(n_kv_heads) * head_size;

    const float* prob_row = probs + static_cast<std::size_t>(h) * seq_len;
    const float* v_head = v_cache + static_cast<std::size_t>(kv_head) * head_size;
    float* out_head = output + static_cast<std::size_t>(h) * head_size;

    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < active_len; t++) {
            acc += prob_row[t] * v_head[t * kv_dim + d];
        }
        out_head[d] = acc;
    }
}


cudaError_t kv_cache_store_cuda(float* cache, const float* current, int kv_dim, int seq_len, int pos)
{
    if (!cache_row_valid(kv_dim, seq_len, pos)) {
        return cudaErrorInvalidValue;
    }
    const int blocks = (kv_dim + kStoreBlock - 1) / kStoreBlock;
    kv_cache_store_kernel<<<blocks, kStoreBlock>>>(cache, current, kv_dim, pos);
    return cudaGetLastError();
}

cudaError_t attention_scores_cuda(
    float* scores,
    const float* q,
    const float* k_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int pos
)
{
    if (!attention_shape_valid(n_heads, n_kv_heads, head_size, seq_len, pos)) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(n_heads, pos + 1);   // one block per (head, visible row)
    attention_scores_kernel<<<grid, kScoreBlock>>>(
        scores, q, k_cache, n_heads, n_kv_heads, head_size, seq_len);
    return cudaGetLastError();
}

cudaError_t softmax_rows_cuda(float* probs, const float* scores, int n_rows, int row_stride, int active_len)
{
    if (n_rows <= 0 || row_stride <= 0 || active_len <= 0 || active_len > row_stride) {
        return cudaErrorInvalidValue;
    }
    softmax_rows_kernel<<<n_rows, kSoftmaxBlock>>>(probs, scores, row_stride, active_len);
    return cudaGetLastError();
}

cudaError_t attention_output_cuda(
    float* output,
    const float* probs,
    const float* v_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int pos
)
{
    if (!attention_shape_valid(n_heads, n_kv_heads, head_size, seq_len, pos)) {
        return cudaErrorInvalidValue;
    }
    attention_output_kernel<<<n_heads, kOutputBlock>>>(
        output, probs, v_cache, n_heads, n_kv_heads, head_size, seq_len, pos + 1);
    return cudaGetLastError();
}

cudaError_t attention_cuda(
    float* output,
    float* scores,
    float* probs,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int pos
)
{
    cudaError_t status = attention_scores_cuda(
        scores, q, k_cache, n_heads, n_kv_heads, head_size, seq_len, pos);
    if (status != cudaSuccess) {
        return status;
    }
    status = softmax_rows_cuda(probs, scores, n_heads, seq_len, pos + 1);
    if (status != cudaSuccess) {
        return status;
    }
    return attention_output_cuda(
        output, probs, v_cache, n_heads, n_kv_heads, head_size, seq_len, pos);
}
