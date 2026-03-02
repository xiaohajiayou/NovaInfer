#pragma once
#include "llaisys/runtime.h"
#include <cstdint>

namespace llaisys::utils {

void nvtx_range_push(const char *name) noexcept;
void nvtx_range_pop() noexcept;
std::uint64_t nvtx_range_start(const char *name) noexcept;
void nvtx_range_end(std::uint64_t id) noexcept;
const char *nvtx_memcpy_tag(llaisysMemcpyKind_t kind, bool is_async) noexcept;

class NvtxScope final {
public:
    explicit NvtxScope(const char *name) noexcept;
    ~NvtxScope();

    NvtxScope(const NvtxScope &) = delete;
    NvtxScope &operator=(const NvtxScope &) = delete;

private:
    std::uint64_t id_{0};
};

} // namespace llaisys::utils

#define LLAISYS_NVTX_CONCAT_INNER(a, b) a##b
#define LLAISYS_NVTX_CONCAT(a, b) LLAISYS_NVTX_CONCAT_INNER(a, b)
#define LLAISYS_NVTX_SCOPE(name) ::llaisys::utils::NvtxScope LLAISYS_NVTX_CONCAT(_llaisys_nvtx_scope_, __LINE__)(name)
