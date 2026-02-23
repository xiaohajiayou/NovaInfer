#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia {

void cuda_check(cudaError_t rc, const char *what, const char *file, int line) {
    if (rc == cudaSuccess) {
        return;
    }
    std::cerr << "[ERROR] CUDA call failed: " << what << ": "
              << cudaGetErrorString(rc) << " at " << file << ":" << line
              << std::endl;
    throw std::runtime_error("CUDA runtime error");
}

} // namespace llaisys::device::nvidia

