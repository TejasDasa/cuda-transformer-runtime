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