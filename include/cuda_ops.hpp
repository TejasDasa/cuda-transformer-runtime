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

#endif