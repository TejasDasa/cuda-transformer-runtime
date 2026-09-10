#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void rmsnorm_kernel(float* output, const float* input, const float* weights, int size, float epsilon)
{
    float summation = 0.0f;

    for (int i = 0; i < size; i++) {
        summation += (input[i] * input[i]);
    }

    float scale = 1.0f / std::sqrt(summation / size + epsilon);

    for (int i = 0; i < size; i++) {
        output[i] = weights[i] * input[i] * scale;
    }
}


__global__ void sum_squares_kernel(float* output, const float* input, int size)
{
    __shared__ float partial[32];

    int i = threadIdx.x;

    if (i < size) {
        partial[i] = input[i] * input[i];
    } else {
        partial[i] = 0;
    }

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
        output[0] = partial[0];
    }
}

int main()
{
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float* d_input = nullptr;

    float weights[] = {1.0f, 1.0f};
    float* d_weights = nullptr;

    float output[2]{};
    float* d_output = nullptr;

    float recovered[2]{};



    // Allocate mem in GPU
    cudaError_t input_status = cudaMalloc(&d_input, sizeof(input));
    if (input_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(input_status) << '\n';
        return 1;
    }

    cudaError_t weights_status = cudaMalloc(&d_weights, sizeof(weights));
    if (weights_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(weights_status) << '\n';
        return 1;
    }

    cudaError_t output_status = cudaMalloc(&d_output, sizeof(output));
    if (output_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(output_status) << '\n';
        return 1;
    }



    // Copy input and weights into GPU mem
    cudaError_t copy_input_status = cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);
    if (copy_input_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(copy_input_status) << '\n';
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);
        return 1;
    }

    cudaError_t copy_weights_status = cudaMemcpy(d_weights, weights, sizeof(weights), cudaMemcpyHostToDevice);
    if (copy_weights_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(copy_weights_status) << '\n';
        cudaFree(d_weights);
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    

    // Run CUDA kernel
    sum_squares_kernel<<<1, 32>>>(d_output, d_input, 8);
    cudaError_t last_error = cudaGetLastError();
    cudaError_t device_sync = cudaDeviceSynchronize();
    if (last_error != cudaSuccess || device_sync != cudaSuccess) {
        std::cerr << cudaGetErrorString(last_error) << '\n';
        std::cerr << cudaGetErrorString(device_sync) << '\n';
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);
        return 1;
    }



    // Recover output
    cudaError_t recover_status = cudaMemcpy(recovered, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    if (recover_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(recover_status) << '\n';
        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);
        return 1;
    }
    std::cout << recovered[0] << '\n';



    // Free allocated memory
    cudaError_t free_input_status = cudaFree(d_input);
    cudaError_t free_weights_status = cudaFree(d_weights);
    cudaError_t free_output_status = cudaFree(d_output);

    if (free_input_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(free_input_status) << '\n';
        return 1;
    }
    
    if (free_weights_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(free_weights_status) << '\n';
        return 1;
    }
    
    if (free_output_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(free_output_status) << '\n';
        return 1;
    }

    return 0;
}