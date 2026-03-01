#include "linear_cuda.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <cstdint>

namespace llaisys::device::nvidia {
void cuda_check(cudaError_t rc, const char *what, const char *file, int line);
} // namespace llaisys::device::nvidia

#define LLAISYS_CUDA_CHECK(call) \
    ::llaisys::device::nvidia::cuda_check((call), #call, __FILE__, __LINE__)

namespace llaisys::ops::cuda {

namespace {

void cublas_check(cublasStatus_t rc, const char *what, const char *file, int line) {
    if (rc != CUBLAS_STATUS_SUCCESS) {
        throw std::invalid_argument(
            std::string("cuBLAS call failed: ") + what + " status=" + std::to_string(static_cast<int>(rc)) +
            " at " + file + ":" + std::to_string(line));
    }
}

#define LLAISYS_CUBLAS_CHECK(call) cublas_check((call), #call, __FILE__, __LINE__)

cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        LLAISYS_CUBLAS_CHECK(cublasCreate(&handle));
    }
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    LLAISYS_CUBLAS_CHECK(cublasSetStream(handle, stream));
    return handle;
}

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, std::int32_t M, std::int32_t N) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                            static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    if (idx >= total) {
        return;
    }
    const std::size_t n = idx % static_cast<std::size_t>(N);
    out[idx] += bias[n];
}

template <>
__global__ void add_bias_kernel<llaisys::fp16_t>(llaisys::fp16_t *out,
                                                 const llaisys::fp16_t *bias,
                                                 std::int32_t M,
                                                 std::int32_t N) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                            static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    if (idx >= total) {
        return;
    }
    const std::size_t n = idx % static_cast<std::size_t>(N);
    __half o = __ushort_as_half(out[idx]._v);
    __half b = __ushort_as_half(bias[n]._v);
    out[idx]._v = __half_as_ushort(__hadd(o, b));
}

template <>
__global__ void add_bias_kernel<llaisys::bf16_t>(llaisys::bf16_t *out,
                                                 const llaisys::bf16_t *bias,
                                                 std::int32_t M,
                                                 std::int32_t N) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                            static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    if (idx >= total) {
        return;
    }
    const std::size_t n = idx % static_cast<std::size_t>(N);
    __nv_bfloat16 o = __ushort_as_bfloat16(out[idx]._v);
    __nv_bfloat16 b = __ushort_as_bfloat16(bias[n]._v);
    out[idx]._v = __bfloat16_as_ushort(__hadd(o, b));
}

template <typename T>
void launch_add_bias(tensor_t out, tensor_t bias, std::int32_t M, std::int32_t N) {
    if (!bias) {
        return;
    }
    constexpr int kBlock = 256;
    const std::size_t total = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    const std::size_t grid = (total + static_cast<std::size_t>(kBlock) - 1) / static_cast<std::size_t>(kBlock);
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    add_bias_kernel<T><<<static_cast<unsigned int>(grid), kBlock, 0, stream>>>(
        reinterpret_cast<T *>(out->data()),
        reinterpret_cast<const T *>(bias->data()),
        M,
        N);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
}

// Compute out[M, N] = in[M, K] * weight[N, K]^T.
// Using cuBLAS column-major convention:
// C_col(NxM) = (W_col)^T (NxK) * A_col(KxM)
// where W_col shares memory with weight row-major [N, K],
// and A_col shares memory with in row-major [M, K].
void gemm_f32(tensor_t out, tensor_t in, tensor_t weight, std::int32_t M, std::int32_t K, std::int32_t N) {
    cublasHandle_t h = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    LLAISYS_CUBLAS_CHECK(cublasSgemm(
        h,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        reinterpret_cast<const float *>(weight->data()),
        K,
        reinterpret_cast<const float *>(in->data()),
        K,
        &beta,
        reinterpret_cast<float *>(out->data()),
        N));
}

void gemm_f16(tensor_t out, tensor_t in, tensor_t weight, std::int32_t M, std::int32_t K, std::int32_t N) {
    cublasHandle_t h = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    LLAISYS_CUBLAS_CHECK(cublasGemmEx(
        h,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        weight->data(),
        CUDA_R_16F,
        K,
        in->data(),
        CUDA_R_16F,
        K,
        &beta,
        out->data(),
        CUDA_R_16F,
        N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_bf16(tensor_t out, tensor_t in, tensor_t weight, std::int32_t M, std::int32_t K, std::int32_t N) {
    cublasHandle_t h = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    LLAISYS_CUBLAS_CHECK(cublasGemmEx(
        h,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        weight->data(),
        CUDA_R_16BF,
        K,
        in->data(),
        CUDA_R_16BF,
        K,
        &beta,
        out->data(),
        CUDA_R_16BF,
        N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const std::int32_t M = static_cast<std::int32_t>(in->shape()[0]);
    const std::int32_t K = static_cast<std::int32_t>(in->shape()[1]);
    const std::int32_t N = static_cast<std::int32_t>(weight->shape()[0]);
    if (M <= 0 || K <= 0 || N <= 0) {
        return;
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        gemm_f32(out, in, weight, M, K, N);
        launch_add_bias<float>(out, bias, M, N);
        return;
    case LLAISYS_DTYPE_F16:
        gemm_f16(out, in, weight, M, K, N);
        launch_add_bias<llaisys::fp16_t>(out, bias, M, N);
        return;
    case LLAISYS_DTYPE_BF16:
        gemm_bf16(out, in, weight, M, K, N);
        launch_add_bias<llaisys::bf16_t>(out, bias, M, N);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops::cuda
