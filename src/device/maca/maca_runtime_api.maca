#include "../runtime_api.hpp"

#include <cuda_runtime_api.h>
#include <iostream>

namespace llaisys::device::nvidia {

void cuda_check(cudaError_t rc, const char *what, const char *file, int line);

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

bool cuda_call_noexcept(cudaError_t rc, const char *what, const char *file, int line) {
    if (rc == cudaSuccess) {
        return true;
    }
    std::cerr << "[ERROR] CUDA runtime API call failed: " << what << ": " << cudaGetErrorString(rc) << " at " << file << ":" << line
              << std::endl;
    return false;
}

#define LLAISYS_CUDA_CALL_NOEXCEPT(call) cuda_call_noexcept((call), #call, __FILE__, __LINE__)

} // namespace

size_t getDeviceFreeMemory(int device_id) {
    size_t free_b = 0;
    size_t total_b = 0;
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaSetDevice(device_id));
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaMemGetInfo(&free_b, &total_b));
    return free_b;
}

size_t getDeviceTotalMemory(int device_id) {
    size_t free_b = 0;
    size_t total_b = 0;
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaSetDevice(device_id));
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaMemGetInfo(&free_b, &total_b));
    return total_b;
}

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaSetDevice(device));
}

void deviceSynchronize() {
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    if (!LLAISYS_CUDA_CALL_NOEXCEPT(cudaStreamCreate(&stream))) {
        return nullptr;
    }
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

llaisysEvent_t createEvent() {
    cudaEvent_t event = nullptr;
    if (!LLAISYS_CUDA_CALL_NOEXCEPT(cudaEventCreateWithFlags(&event, cudaEventDisableTiming))) {
        return nullptr;
    }
    return reinterpret_cast<llaisysEvent_t>(event);
}

void destroyEvent(llaisysEvent_t event) {
    if (event == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
}

void eventRecord(llaisysEvent_t event, llaisysStream_t stream) {
    if (event == nullptr || stream == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(
        cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(stream)));
}

void streamWaitEvent(llaisysStream_t stream, llaisysEvent_t event) {
    if (stream == nullptr || event == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(
        cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<cudaEvent_t>(event), 0));
}

void eventSynchronize(llaisysEvent_t event) {
    if (event == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    if (!LLAISYS_CUDA_CALL_NOEXCEPT(cudaMalloc(&ptr, size))) {
        return nullptr;
    }
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    if (!LLAISYS_CUDA_CALL_NOEXCEPT(cudaMallocHost(&ptr, size))) {
        return nullptr;
    }
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    LLAISYS_NVTX_SCOPE(llaisys::utils::nvtx_memcpy_tag(kind, false));
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    LLAISYS_NVTX_SCOPE(llaisys::utils::nvtx_memcpy_tag(kind, true));
    if (stream == nullptr) {
        (void)LLAISYS_CUDA_CALL_NOEXCEPT(cudaMemcpy(dst, src, size, to_cuda_memcpy_kind(kind)));
        return;
    }
    (void)LLAISYS_CUDA_CALL_NOEXCEPT(
        cudaMemcpyAsync(dst, src, size, to_cuda_memcpy_kind(kind), reinterpret_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &createEvent,
    &destroyEvent,
    &eventRecord,
    &streamWaitEvent,
    &eventSynchronize,
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
