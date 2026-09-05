#include <cuda_runtime.h>
#include <iostream>


int main(void)
{
    int device_count = 0;

    cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status != cudaSuccess) {
        std::cerr << cudaGetErrorString(status) << '\n';
        return 1;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;

        cudaError_t prop_status = cudaGetDeviceProperties(&prop, i);

        if (prop_status != cudaSuccess) {
            std::cerr << cudaGetErrorString(prop_status) << '\n';
            continue;
        }

        double global_mem_gib = static_cast<double>(prop.totalGlobalMem) / (1024 * 1024 * 1024);
        double shared_mem_kib = static_cast<double>(prop.sharedMemPerBlock) / 1024;

        std::cout << "Device " << i << '\n';
        std::cout << "  Name: " << prop.name << '\n';
        std::cout << "  Compute Capability: " << prop.major << '.' << prop.minor << '\n';
        std::cout << "  multiProcessorCount: " << prop.multiProcessorCount << '\n';
        std::cout << "  warpSize: " << prop.warpSize << '\n';
        std::cout << "  maxThreadsPerBlock: " << prop.maxThreadsPerBlock << '\n';
        std::cout << "  totalGlobalMem: " << global_mem_gib << " GiB" << '\n';
        std::cout << "  sharedMemPerBlock: " << shared_mem_kib << " KiB" << '\n';
    }

    return 0;
}