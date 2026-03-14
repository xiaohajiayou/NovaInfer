#pragma once
#include "llaisys/runtime.h"

#include "../utils.hpp"

namespace llaisys::device {
const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type);
size_t getDeviceFreeMemory(llaisysDeviceType_t device_type, int device_id);
size_t getDeviceTotalMemory(llaisysDeviceType_t device_type, int device_id);

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {
const LlaisysRuntimeAPI *getRuntimeAPI();
}

#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const LlaisysRuntimeAPI *getRuntimeAPI();
size_t getDeviceFreeMemory(int device_id);
size_t getDeviceTotalMemory(int device_id);
}
#endif
} // namespace llaisys::device
