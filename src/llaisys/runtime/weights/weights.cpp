#include "weights.hpp"

#include "../../../utils/check.hpp"

#include <unordered_set>

namespace llaisys::runtime::weights {

void replace_slot(llaisysTensor_t *slot, llaisysTensor_t new_handle) {
    CHECK_ARGUMENT(slot != nullptr, "weights: slot must not be null");
    if (*slot == new_handle) {
        return;
    }
    if (*slot != nullptr) {
        tensorDestroy(*slot);
    }
    *slot = new_handle;
}

void destroy_unique(const std::vector<llaisysTensor_t *> &slots) {
    std::unordered_set<const void *> seen{};
    for (llaisysTensor_t *slot : slots) {
        if (slot == nullptr || *slot == nullptr) {
            continue;
        }
        const void *key = static_cast<const void *>(*slot);
        if (seen.insert(key).second) {
            tensorDestroy(*slot);
        }
        *slot = nullptr;
    }
}

} // namespace llaisys::runtime::weights
