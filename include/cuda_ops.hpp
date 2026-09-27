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

#endif
