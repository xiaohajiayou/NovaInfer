#include "nvtx.hpp"
#include <cstdint>

#ifdef ENABLE_NVIDIA_API
extern "C" int nvtxRangePushA(const char *message);
extern "C" int nvtxRangePop(void);
extern "C" std::uint64_t nvtxRangeStartA(const char *message);
extern "C" int nvtxRangeEnd(std::uint64_t id);
#endif

namespace llaisys::utils {

void nvtx_range_push(const char *name) noexcept {
#ifdef ENABLE_NVIDIA_API
    if (name == nullptr || *name == '\0') {
        return;
    }
    (void)nvtxRangePushA(name);
#else
    (void)name;
#endif
}

void nvtx_range_pop() noexcept {
#ifdef ENABLE_NVIDIA_API
    (void)nvtxRangePop();
#endif
}

std::uint64_t nvtx_range_start(const char *name) noexcept {
#ifdef ENABLE_NVIDIA_API
    if (name == nullptr || *name == '\0') {
        return 0;
    }
    return nvtxRangeStartA(name);
#else
    (void)name;
    return 0;
#endif
}

void nvtx_range_end(std::uint64_t id) noexcept {
#ifdef ENABLE_NVIDIA_API
    if (id == 0) {
        return;
    }
    (void)nvtxRangeEnd(id);
#else
    (void)id;
#endif
}

const char *nvtx_memcpy_tag(llaisysMemcpyKind_t kind, bool is_async) noexcept {
    if (is_async) {
        switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            return "memcpy/async/h2h";
        case LLAISYS_MEMCPY_H2D:
            return "memcpy/async/h2d";
        case LLAISYS_MEMCPY_D2H:
            return "memcpy/async/d2h";
        case LLAISYS_MEMCPY_D2D:
            return "memcpy/async/d2d";
        default:
            return "memcpy/async/unknown";
        }
    }
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return "memcpy/sync/h2h";
    case LLAISYS_MEMCPY_H2D:
        return "memcpy/sync/h2d";
    case LLAISYS_MEMCPY_D2H:
        return "memcpy/sync/d2h";
    case LLAISYS_MEMCPY_D2D:
        return "memcpy/sync/d2d";
    default:
        return "memcpy/sync/unknown";
    }
}

NvtxScope::NvtxScope(const char *name) noexcept {
    id_ = nvtx_range_start(name);
}

NvtxScope::~NvtxScope() {
    nvtx_range_end(id_);
}

} // namespace llaisys::utils
