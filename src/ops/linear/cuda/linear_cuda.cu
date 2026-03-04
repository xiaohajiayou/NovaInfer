#include "linear_cuda.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <array>
#include <cstdint>
#include <unordered_map>

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

cublasLtHandle_t get_cublaslt_handle() {
    static thread_local cublasLtHandle_t handle = nullptr;
    if (handle == nullptr) {
        LLAISYS_CUBLAS_CHECK(cublasLtCreate(&handle));
    }
    return handle;
}

cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        LLAISYS_CUBLAS_CHECK(cublasCreate(&handle));
    }
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    LLAISYS_CUBLAS_CHECK(cublasSetStream(handle, stream));
    return handle;
}

cudaDataType_t to_cuda_data_type(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

cublasComputeType_t to_cublas_compute_type(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUBLAS_COMPUTE_32F;
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return CUBLAS_COMPUTE_32F;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

template <typename T, cublasStatus_t (*DestroyFn)(T)>
class CublasLtScopedHandle {
public:
    CublasLtScopedHandle() = default;
    ~CublasLtScopedHandle() {
        if (handle_ != nullptr) {
            (void)DestroyFn(handle_);
        }
    }

    T *out_ptr() { return &handle_; }
    T get() const { return handle_; }

private:
    T handle_{nullptr};
};

struct LtAlgoKey {
    int dtype{};
    int32_t m{};
    int32_t n{};
    int32_t k{};
    int32_t device{};

    bool operator==(const LtAlgoKey &other) const noexcept {
        return dtype == other.dtype && m == other.m && n == other.n && k == other.k && device == other.device;
    }
};

struct LtAlgoKeyHash {
    std::size_t operator()(const LtAlgoKey &key) const noexcept {
        std::size_t h = 0;
        const auto mix = [&h](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        };
        mix(static_cast<std::size_t>(key.dtype));
        mix(static_cast<std::size_t>(key.m));
        mix(static_cast<std::size_t>(key.n));
        mix(static_cast<std::size_t>(key.k));
        mix(static_cast<std::size_t>(key.device));
        return h;
    }
};

struct LtAlgoEntry {
    cublasLtMatmulAlgo_t algo{};
    std::size_t workspace_size{0};
};

std::unordered_map<LtAlgoKey, LtAlgoEntry, LtAlgoKeyHash> &lt_algo_cache() {
    static thread_local std::unordered_map<LtAlgoKey, LtAlgoEntry, LtAlgoKeyHash> cache;
    return cache;
}

struct LtWorkspace {
    ~LtWorkspace() {
        if (ptr != nullptr) {
            (void)cudaFree(ptr);
            ptr = nullptr;
            size = 0;
            device = -1;
        }
    }

    void *ensure(std::size_t need, int dev) {
        if (need == 0) {
            return nullptr;
        }
        if (device != dev) {
            if (ptr != nullptr) {
                LLAISYS_CUDA_CHECK(cudaFree(ptr));
                ptr = nullptr;
                size = 0;
            }
            device = dev;
        }
        if (size < need) {
            if (ptr != nullptr) {
                LLAISYS_CUDA_CHECK(cudaFree(ptr));
                ptr = nullptr;
                size = 0;
            }
            LLAISYS_CUDA_CHECK(cudaSetDevice(dev));
            LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, need));
            size = need;
        }
        return ptr;
    }

    void *ptr{nullptr};
    std::size_t size{0};
    int device{-1};
};

LtWorkspace &lt_workspace_buffer() {
    static thread_local LtWorkspace buf;
    return buf;
}

constexpr std::size_t kLtWorkspaceLimit = 64ULL * 1024ULL * 1024ULL;

LtAlgoEntry select_lt_algo(cublasLtHandle_t lt,
                           cublasLtMatmulDesc_t op_desc,
                           cublasLtMatrixLayout_t a_desc,
                           cublasLtMatrixLayout_t b_desc,
                           cublasLtMatrixLayout_t c_desc,
                           int32_t m,
                           int32_t n,
                           int32_t k,
                           llaisysDataType_t dtype,
                           int32_t device) {
    LtAlgoKey key{
        static_cast<int>(dtype),
        m,
        n,
        k,
        device};
    auto &cache = lt_algo_cache();
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    CublasLtScopedHandle<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy> pref;
    LLAISYS_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(pref.out_ptr()));

    const std::size_t workspace_limit = kLtWorkspaceLimit;
    LLAISYS_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref.get(),
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit,
        sizeof(workspace_limit)));

    constexpr int kMaxAlgos = 16;
    std::array<cublasLtMatmulHeuristicResult_t, kMaxAlgos> heuristics{};
    int returned_results = 0;
    LLAISYS_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        pref.get(),
        kMaxAlgos,
        heuristics.data(),
        &returned_results));
    if (returned_results <= 0) {
        throw std::invalid_argument("cublasLtMatmulAlgoGetHeuristic returned no algorithm");
    }

    LtAlgoEntry entry{};
    entry.algo = heuristics[0].algo;
    entry.workspace_size = heuristics[0].workspaceSize;
    cache.emplace(key, entry);
    return entry;
}

void gemm_bias_fused_lt(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, std::int32_t M, std::int32_t K, std::int32_t N) {
    if (bias == nullptr) {
        throw std::invalid_argument("linear cuda fused path requires non-null bias");
    }
    if (bias->ndim() != 1 || static_cast<std::int32_t>(bias->shape()[0]) != N) {
        throw std::invalid_argument("linear cuda fused path bias shape mismatch");
    }
    if (bias->dtype() != out->dtype() || bias->deviceType() != out->deviceType() || bias->deviceId() != out->deviceId()) {
        throw std::invalid_argument("linear cuda fused path bias tensor mismatch");
    }

    const auto data_type = to_cuda_data_type(out->dtype());
    const auto compute_type = to_cublas_compute_type(out->dtype());
    cublasLtHandle_t lt = get_cublaslt_handle();
    auto stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    CublasLtScopedHandle<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy> op_desc;
    LLAISYS_CUBLAS_CHECK(cublasLtMatmulDescCreate(op_desc.out_ptr(), compute_type, CUDA_R_32F));

    const cublasOperation_t op_a = CUBLAS_OP_T;
    const cublasOperation_t op_b = CUBLAS_OP_N;
    LLAISYS_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(op_desc.get(), CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    LLAISYS_CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(op_desc.get(), CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    LLAISYS_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc.get(),
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue)));
    const void *bias_ptr = bias->data();
    LLAISYS_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc.get(),
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_ptr,
        sizeof(bias_ptr)));

    CublasLtScopedHandle<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> a_desc;
    CublasLtScopedHandle<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> b_desc;
    CublasLtScopedHandle<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> c_desc;

    // A: weight as column-major [K, N], op(A)=T => [N, K]
    LLAISYS_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(a_desc.out_ptr(), data_type, K, N, K));
    // B: input as column-major [K, M], op(B)=N => [K, M]
    LLAISYS_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(b_desc.out_ptr(), data_type, K, M, K));
    // C/D: output as column-major [N, M]
    LLAISYS_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(c_desc.out_ptr(), data_type, N, M, N));

    const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    LLAISYS_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(a_desc.get(), CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    LLAISYS_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(b_desc.get(), CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    LLAISYS_CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(c_desc.get(), CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    const LtAlgoEntry algo = select_lt_algo(
        lt,
        op_desc.get(),
        a_desc.get(),
        b_desc.get(),
        c_desc.get(),
        N,
        M,
        K,
        out->dtype(),
        out->deviceId());

    void *workspace_ptr = nullptr;
    std::size_t workspace_size = algo.workspace_size;
    if (workspace_size > 0) {
        auto &workspace = lt_workspace_buffer();
        workspace_ptr = workspace.ensure(workspace_size, out->deviceId());
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t rc = cublasLtMatmul(
        lt,
        op_desc.get(),
        &alpha,
        weight->data(),
        a_desc.get(),
        in->data(),
        b_desc.get(),
        &beta,
        out->data(),
        c_desc.get(),
        out->data(),
        c_desc.get(),
        &algo.algo,
        workspace_ptr,
        workspace_size,
        stream);
    LLAISYS_CUBLAS_CHECK(rc);
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
        if (bias != nullptr) {
            gemm_bias_fused_lt(out, in, weight, bias, M, K, N);
            return;
        }
        gemm_f32(out, in, weight, M, K, N);
        return;
    case LLAISYS_DTYPE_F16:
        if (bias != nullptr) {
            gemm_bias_fused_lt(out, in, weight, bias, M, K, N);
            return;
        }
        gemm_f16(out, in, weight, M, K, N);
        return;
    case LLAISYS_DTYPE_BF16:
        if (bias != nullptr) {
            gemm_bias_fused_lt(out, in, weight, bias, M, K, N);
            return;
        }
        gemm_bf16(out, in, weight, M, K, N);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops::cuda
