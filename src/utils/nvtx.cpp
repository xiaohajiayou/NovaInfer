#include "nvtx.hpp"

#ifdef ENABLE_NVIDIA_API
extern "C" int nvtxRangePushA(const char *message);
extern "C" int nvtxRangePop(void);
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
#ifdef ENABLE_NVIDIA_API
    if (name == nullptr || *name == '\0') {
        return;
    }
    const int rc = nvtxRangePushA(name);
    pushed_ = (rc >= 0);
#else
    (void)name;
#endif
}

NvtxScope::~NvtxScope() {
    if (!pushed_) {
        return;
    }
    nvtx_range_pop();
}

} // namespace llaisys::utils
