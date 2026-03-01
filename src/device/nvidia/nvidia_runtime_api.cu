#include "../runtime_api.hpp"

#include <cuda_runtime_api.h>

namespace llaisys::device::nvidia {

void cuda_check(cudaError_t rc, const char *what, const char *file, int line);

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace {

cudaMemcpyKind to_cuda_memcpy_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        CHECK_ARGUMENT(false, "nvidia memcpy: unsupported memcpy kind");
    }
}

} // namespace

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    LLAISYS_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    LLAISYS_CUDA_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    LLAISYS_CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    LLAISYS_CUDA_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    LLAISYS_CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    LLAISYS_CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    LLAISYS_CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    LLAISYS_CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    LLAISYS_CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    LLAISYS_NVTX_SCOPE(llaisys::utils::nvtx_memcpy_tag(kind, false));
    LLAISYS_CUDA_CHECK(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    LLAISYS_NVTX_SCOPE(llaisys::utils::nvtx_memcpy_tag(kind, true));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(dst,
                                       src,
                                       size,
                                       to_cuda_memcpy_kind(kind),
                                       reinterpret_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia
