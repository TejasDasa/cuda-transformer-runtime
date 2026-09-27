#ifndef DEVICE_BUFFER_HPP
#define DEVICE_BUFFER_HPP

// Small RAII helpers shared by the CUDA tests.  Header-only so no extra
// source file needs linking.

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// Throws with a readable message if a CUDA runtime call failed.
inline void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status)
        );
    }
}

// Owns one cudaMalloc'd float buffer.  The destructor frees it, so every
// allocation is released whether we return normally, return early on a
// failed check, or unwind through an exception.  Not copyable: two owners
// of one device pointer would double-free.
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
        : count_(count)
    {
        check_cuda(cudaMalloc(&ptr_, bytes()), "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        // cudaFree cannot throw and a destructor must not, so the status
        // is intentionally ignored here.
        cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() { return ptr_; }
    const float* data() const { return ptr_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(float); }

    // Host -> device.  `source` must point at count() floats.
    void upload(const float* source, const char* what)
    {
        check_cuda(
            cudaMemcpy(ptr_, source, bytes(), cudaMemcpyHostToDevice),
            what
        );
    }

    // Device -> host into a freshly sized vector.
    std::vector<float> download(const char* what) const
    {
        std::vector<float> host(count_);
        check_cuda(
            cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost),
            what
        );
        return host;
    }

private:
    float* ptr_ = nullptr;
    std::size_t count_ = 0;
};

#endif
