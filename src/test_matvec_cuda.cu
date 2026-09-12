#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#include "cpu_ops.hpp"
#include "cuda_ops.hpp"


int main()
{
    constexpr int rows = 7;
    constexpr int cols = 289;

    float matrix[rows*cols]{};
    float input[cols]{};
    float recovered[rows]{};
    float output[rows]{};

    float* d_input = nullptr;
    float* d_matrix = nullptr;
    float* d_output = nullptr;


    for (int j = 0; j < cols; j++) {
        input[j] = static_cast<float>((j % 17) - 8) * 0.1f;
    }

    for (int k = 0; k < rows * cols; k++) {
        matrix[k] = static_cast<float>((k % 13) - 6) * 0.05f;
    }

    
    cudaError_t input_alloc_status = cudaMalloc(&d_input, sizeof(input));
    if (input_alloc_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(input_alloc_status) << '\n';
        return 1;
    }

    cudaError_t matrix_alloc_status = cudaMalloc(&d_matrix, sizeof(matrix));
    if (matrix_alloc_status !=  cudaSuccess) {
        std::cerr << cudaGetErrorString(matrix_alloc_status) << '\n';
        cudaFree(d_input);
        return 1;
    }

    cudaError_t output_status = cudaMalloc(&d_output, sizeof(output));
    if (output_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(output_status) << '\n';
        cudaFree(d_input);
        cudaFree(d_matrix);
        return 1;
    }



    cudaError_t input_copy_status = cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);
    if (input_copy_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(input_copy_status) << '\n';
        cudaFree(d_input);
        cudaFree(d_matrix);
        cudaFree(d_output);
        return 1;
    }

    cudaError_t matrix_copy_status = cudaMemcpy(d_matrix, matrix, sizeof(matrix), cudaMemcpyHostToDevice);
    if (matrix_copy_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(matrix_copy_status) << '\n';
        cudaFree(d_input);
        cudaFree(d_matrix);
        cudaFree(d_output);
        return 1;
    }



    
    cudaError_t kernel_error = matvec_cuda(d_output, d_matrix, d_input, rows, cols);
    cudaError_t device_sync = cudaDeviceSynchronize();
    if (kernel_error != cudaSuccess || device_sync != cudaSuccess) {
        std::cerr << cudaGetErrorString(kernel_error) << '\n';
        std::cerr << cudaGetErrorString(device_sync) << '\n';
        cudaFree(d_input);
        cudaFree(d_matrix);
        cudaFree(d_output);
        return 1;
    }



    cudaError_t recover_status = cudaMemcpy(recovered, d_output, sizeof(recovered), cudaMemcpyDeviceToHost);
    if (recover_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(recover_status);
        cudaFree(d_input);
        cudaFree(d_matrix);
        cudaFree(d_output);
        return 1;
    }



    for (int i = 0; i < rows; i++) {
        std::cout << recovered[i] << ", \n";
    }



    matvec(output, matrix, input, rows, cols);

    bool all_passed = true;
    float max_absolute_error = 0;

    for (int i = 0; i < rows; i++) {
        if (!std::isfinite(output[i]) || !std::isfinite(recovered[i])) {
            all_passed = false;
            std::cout << "Nonfinite result at index " << i
                    << ", CPU: " << output[i]
                    << ", GPU: " << recovered[i] << '\n';
            continue;
        }

        float abs_diff = std::abs(output[i] - recovered[i]);

        if (abs_diff > max_absolute_error) {
            max_absolute_error = abs_diff;
        }

        float diff_allowed = (std::abs(output[i]) * 1e-5f) + 1e-5f;

        if (abs_diff > diff_allowed) {
            all_passed = false;
            std::cout << "Index: " << i << ", CPU: " << output[i] << ", GPU: " << recovered[i] << '\n';
        }
    }

    if (all_passed) {
        std::cout << "All Passed\n";
        std::cout << "Max error: " << max_absolute_error << '\n';
    }


    cudaError_t free_input_status = cudaFree(d_input);
    cudaError_t free_matrix_status = cudaFree(d_matrix);
    cudaError_t free_output_status = cudaFree(d_output);

    if (free_input_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(free_input_status) << '\n';
        return 1;
    }
    
    if (free_matrix_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(free_matrix_status) << '\n';
        return 1;
    }
    
    if (free_output_status != cudaSuccess) {
        std::cerr << cudaGetErrorString(free_output_status) << '\n';
        return 1;
    }

    return 0;
}