#pragma once
#include "llaisys/runtime.h"

namespace llaisys::utils {

void nvtx_range_push(const char *name) noexcept;
void nvtx_range_pop() noexcept;
const char *nvtx_memcpy_tag(llaisysMemcpyKind_t kind, bool is_async) noexcept;

class NvtxScope final {
public:
    explicit NvtxScope(const char *name) noexcept;
    ~NvtxScope();

    NvtxScope(const NvtxScope &) = delete;
    NvtxScope &operator=(const NvtxScope &) = delete;

private:
    bool pushed_{false};
};

} // namespace llaisys::utils

#define LLAISYS_NVTX_CONCAT_INNER(a, b) a##b
#define LLAISYS_NVTX_CONCAT(a, b) LLAISYS_NVTX_CONCAT_INNER(a, b)
#define LLAISYS_NVTX_SCOPE(name) ::llaisys::utils::NvtxScope LLAISYS_NVTX_CONCAT(_llaisys_nvtx_scope_, __LINE__)(name)
