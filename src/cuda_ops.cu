#include <cuda_runtime.h>
#include <cmath>

#include "cuda_ops.hpp"

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
