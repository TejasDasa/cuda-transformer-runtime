#include "checkpoint.hpp"
#include "cpu_ops.hpp"
#include "cuda_ops.hpp"

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <exception>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Please provide a checkpoint path\n";
        return 1;
    }

    try {
        Checkpoint checkpoint{argv[1]};
        const auto& config = checkpoint.config();
        const auto& weights = checkpoint.weights();

        std::int64_t vocab_size = config.vocab_size;
        if (vocab_size < 0) {
            vocab_size = -vocab_size;
        }
        const auto vocab_count = static_cast<std::uint64_t>(vocab_size);

        int token_id = 42;
        if (token_id < 0 || static_cast<uint64_t>(token_id) >= vocab_count) {
            std::cerr << "Error: Token out of vocab range\n";
            return 1;
        }

        std::vector<float> x(config.dim);
        const float* row = weights.token_embedding_table + (config.dim * static_cast<std::size_t>(token_id));
        for (int i = 0; i < config.dim; i++) {
            x[i] = row[i];
        }   

        std::vector<float> normalized_cpu(config.dim);
        std::vector<float> q_cpu(config.dim);

        rmsnorm(normalized_cpu.data(), x.data(), weights.rms_att_weight, config.dim, 1e-5f);
        matvec(q_cpu.data(), weights.wq, normalized_cpu.data(), config.dim, config.dim);



        float* d_x = nullptr;
        float* d_rms_weights = nullptr;
        float* d_wq = nullptr;
        float* d_normalized = nullptr;
        float* d_q = nullptr;

        const std::size_t vector_bytes = static_cast<std::size_t>(config.dim) * sizeof(float);
        const std::size_t matrix_bytes = static_cast<std::size_t>(config.dim) * config.dim * sizeof(float);


        cudaError_t x_malloc_status = cudaMalloc(&d_x, vector_bytes);
        if (x_malloc_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(x_malloc_status) << '\n';
            return 1;
        }

        cudaError_t rms_malloc_status = cudaMalloc(&d_rms_weights, vector_bytes);
        if (rms_malloc_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(rms_malloc_status) << '\n';
            cudaFree(d_x);
            return 1;
        }

        cudaError_t wq_malloc_status = cudaMalloc(&d_wq, matrix_bytes);
        if (wq_malloc_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(wq_malloc_status) << '\n';
            cudaFree(d_x);
            cudaFree(d_rms_weights);
            return 1;
        }

        cudaError_t normalized_malloc_status = cudaMalloc(&d_normalized, vector_bytes);
        if (normalized_malloc_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(normalized_malloc_status) << '\n';
            cudaFree(d_x);
            cudaFree(d_rms_weights);
            cudaFree(d_wq);
            return 1;
        }

        cudaError_t q_malloc_status = cudaMalloc(&d_q, vector_bytes);
        if (q_malloc_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(q_malloc_status) << '\n';
            cudaFree(d_x);
            cudaFree(d_rms_weights);
            cudaFree(d_wq);
            cudaFree(d_normalized);
            return 1;
        }



        cudaError_t x_copy_status = cudaMemcpy(d_x, x.data(), vector_bytes, cudaMemcpyHostToDevice);
        if (x_copy_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(x_copy_status) << '\n';
            cudaFree(d_x);
            cudaFree(d_rms_weights);
            cudaFree(d_wq);
            cudaFree(d_normalized);
            cudaFree(d_q);
            return 1;
        }

        cudaError_t rms_copy_status = cudaMemcpy(d_rms_weights, weights.rms_att_weight, vector_bytes, cudaMemcpyHostToDevice);
        if (rms_copy_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(rms_copy_status) << '\n';
            cudaFree(d_x);
            cudaFree(d_rms_weights);
            cudaFree(d_wq);
            cudaFree(d_normalized);
            cudaFree(d_q);
            return 1;
        }

        cudaError_t wq_copy_status = cudaMemcpy(d_wq, weights.wq, matrix_bytes, cudaMemcpyHostToDevice);
        if (wq_copy_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(wq_copy_status) << '\n';
            cudaFree(d_x);
            cudaFree(d_rms_weights);
            cudaFree(d_wq);
            cudaFree(d_normalized);
            cudaFree(d_q);
            return 1;
        }
    }

    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    
    return 0;
}