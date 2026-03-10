#include "llaisys/runtime.h"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"
#include "../utils/nvtx.hpp"

// Llaisys API for setting context runtime.
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
    llaisys::core::context().setDevice(device_type, device_id);
}

// Llaisys API for getting the runtime APIs
__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    return llaisys::device::getRuntimeAPI(device_type);
}

__C void llaisysNvtxRangePush(const char *name) {
    llaisys::utils::nvtx_range_push(name);
}

__C void llaisysNvtxRangePop(void) {
    llaisys::utils::nvtx_range_pop();
}
