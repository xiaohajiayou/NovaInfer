#include "context.hpp"
#include "../../utils.hpp"
#include <thread>

namespace llaisys::core {

Context::Context() : _current_runtime(nullptr) {
    // All device types, put CPU at the end
    std::vector<llaisysDeviceType_t> device_typs;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
    }
    device_typs.push_back(LLAISYS_DEVICE_CPU);

    // Create runtimes for each device type.
    // Activate the first available device. If no other device is available, activate CPU runtime.
    for (auto device_type : device_typs) {
        const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);
        int device_count = api_->get_device_count();
        std::vector<Runtime *> runtimes_(device_count);
        for (int device_id = 0; device_id < device_count; device_id++) {

            if (_current_runtime == nullptr) {
                auto runtime = new Runtime(device_type, device_id);
                runtime->_activate();
                runtimes_[device_id] = runtime;
                _current_runtime = runtime;
            }
        }
        _runtime_map[device_type] = runtimes_;
    }
}

Context::~Context() {
    // Destroy current runtime first.
    delete _current_runtime;

    for (auto &runtime_entry : _runtime_map) {
        std::vector<Runtime *> runtimes = runtime_entry.second;
        for (auto runtime : runtimes) {
            if (runtime != nullptr && runtime != _current_runtime) {
                runtime->_activate();
                delete runtime;
            }
        }
        runtimes.clear();
    }
    _current_runtime = nullptr;
    _runtime_map.clear();
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    // If doest not match the current runtime.
    if (_current_runtime == nullptr || _current_runtime->deviceType() != device_type || _current_runtime->deviceId() != device_id) {
        auto &runtimes = _runtime_map[device_type];
        CHECK_ARGUMENT((size_t)device_id < runtimes.size() && device_id >= 0, "invalid device id");
        if (_current_runtime != nullptr) {
            _current_runtime->_deactivate();
        }
        if (runtimes[device_id] == nullptr) {
            runtimes[device_id] = new Runtime(device_type, device_id);
        }
        runtimes[device_id]->_activate();
        _current_runtime = runtimes[device_id];
    }
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

// Global API to get thread-local context.
Context &context() {
    // NOTE:
    // Storage holds Runtime&. In multi-thread inference, tensors can be allocated
    // on worker threads but destroyed later on a different thread. If thread-local
    // Context is destructed at worker thread exit, those Runtime references become
    // dangling and model destroy can crash.
    //
    // Keep per-thread Context alive for process lifetime to avoid Runtime
    // use-after-free across thread boundaries. This is a bounded leak (one Context
    // per thread touching core runtime), acceptable for current stage.
    thread_local Context *thread_context = new Context();
    return *thread_context;
}

} // namespace llaisys::core
