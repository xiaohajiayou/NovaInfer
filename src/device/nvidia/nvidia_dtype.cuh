#pragma once

#include "../../utils/types.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llaisys::device::nvidia::dtype {

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float to_float<llaisys::fp16_t>(llaisys::fp16_t v) {
    return __half2float(__ushort_as_half(v._v));
}

template <>
__device__ __forceinline__ float to_float<llaisys::bf16_t>(llaisys::bf16_t v) {
    return __bfloat162float(__ushort_as_bfloat16(v._v));
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ llaisys::fp16_t from_float<llaisys::fp16_t>(float v) {
    const __half h = __float2half_rn(v);
    return llaisys::fp16_t{__half_as_ushort(h)};
}

template <>
__device__ __forceinline__ llaisys::bf16_t from_float<llaisys::bf16_t>(float v) {
    const __nv_bfloat16 b = __float2bfloat16(v);
    return llaisys::bf16_t{__bfloat16_as_ushort(b)};
}

} // namespace llaisys::device::nvidia::dtype
