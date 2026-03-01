#include "llaisys/models/qwen2.h"

#include "qwen2_model.hpp"

#include "../../utils/check.hpp"

#include <exception>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

namespace llaisys::models::qwen2 {

struct LlaisysQwen2ModelImpl {
    explicit LlaisysQwen2ModelImpl(const LlaisysQwen2Meta &meta,
                                   llaisysDeviceType_t device,
                                   int *device_ids,
                                   int ndevice)
        : model(std::make_unique<Qwen2Model>(meta, device, device_ids, ndevice)) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2: failed to allocate model");
        const int rc = model->configure_runtime(
            llaisys::runtime::kv_cache::KvCacheLayout::BLOCK,
            16,
            static_cast<size_t>(meta.maxseq),
            static_cast<int64_t>(meta.maxseq));
        CHECK_ARGUMENT(rc == 0, "Qwen2: failed to configure runtime");
    }

    std::unique_ptr<Qwen2Model> model;
};

} // namespace llaisys::models::qwen2

__C {

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::qwen2::LlaisysQwen2ModelImpl> impl;
};

static void fail_fast(const char *where, const std::exception &e) {
    std::cerr << "[FATAL] Qwen2 C API failure in " << where << ": " << e.what() << std::endl;
    std::abort();
}

static void fail_fast_unknown(const char *where) {
    std::cerr << "[FATAL] Qwen2 C API unknown failure in " << where << std::endl;
    std::abort();
}

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                           llaisysDeviceType_t device,
                                                           int *device_ids,
                                                           int ndevice) {
    try {
        CHECK_ARGUMENT(meta != nullptr, "Qwen2: meta must not be null");
        auto *handle = new LlaisysQwen2Model{};
        handle->impl = std::make_unique<llaisys::models::qwen2::LlaisysQwen2ModelImpl>(*meta, device, device_ids, ndevice);
        return handle;
    } catch (const std::exception &e) {
        fail_fast("llaisysQwen2ModelCreate", e);
    } catch (...) {
        fail_fast_unknown("llaisysQwen2ModelCreate");
    }
    return nullptr;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }

    try {
        delete model;
    } catch (const std::exception &e) {
        fail_fast("llaisysQwen2ModelDestroy", e);
    } catch (...) {
        fail_fast_unknown("llaisysQwen2ModelDestroy");
    }
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    try {
        CHECK_ARGUMENT(model != nullptr && model->impl != nullptr, "Qwen2: model must not be null");
        return model->impl->model->weights();
    } catch (const std::exception &e) {
        fail_fast("llaisysQwen2ModelWeights", e);
    } catch (...) {
        fail_fast_unknown("llaisysQwen2ModelWeights");
    }
    return nullptr;
}

} // extern "C"
