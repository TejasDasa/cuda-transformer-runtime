#ifndef CUDA_OPS_HPP
#define CUDA_OPS_HPP

#include <cuda_runtime.h>

cudaError_t matvec_cuda(
    float* output, 
    const float* matrix, 
    const float* input, 
    int rows, 
    int cols
);

cudaError_t rmsnorm_cuda(
    float* output, 
    const float* input, 
    const float* weights, 
    int size, 
    float epsilon
);

// In-place RoPE on a device vector of n_heads * head_size floats.
// cos_row / sin_row are device pointers to the head_size / 2 entries for
// one token position.  Same semantics as the CPU rope().  Returns the
// launch status only; the caller synchronises.
cudaError_t rope_cuda(
    float* vec,
    int n_heads,
    int head_size,
    const float* cos_row,
    const float* sin_row
);

// ---------------------------------------------------------------------
// KV cache and single-token causal attention (device pointers only).
// Same shapes and semantics as the CPU versions in cpu_ops.hpp.  Every
// wrapper returns the launch status and never synchronises; invalid
// shapes return cudaErrorInvalidValue without launching anything.
// ---------------------------------------------------------------------

// cache[pos, :] = current   (cache is [seq_len, kv_dim])
cudaError_t kv_cache_store_cuda(
    float* cache,
    const float* current,
    int kv_dim,
    int seq_len,
    int pos
);

// scores[h, t] = dot(q[h], k_cache[t, kv_head(h)]) / sqrt(head_size)
// for t in [0, pos].  scores is [n_heads, seq_len].
cudaError_t attention_scores_cuda(
    float* scores,
    const float* q,
    const float* k_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int pos
);

// Row-wise stable softmax over the first active_len entries of each of
// n_rows rows; both buffers are [n_rows, row_stride].
cudaError_t softmax_rows_cuda(
    float* probs,
    const float* scores,
    int n_rows,
    int row_stride,
    int active_len
);

// output[h, d] = sum_{t <= pos} probs[h, t] * v_cache[t, kv_head(h), d]
cudaError_t attention_output_cuda(
    float* output,
    const float* probs,
    const float* v_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int pos
);

// Convenience: the three stages above in order on the default stream.
// Returns the first launch error encountered.
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
);

#endif
