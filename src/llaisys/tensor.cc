#include "llaisys_tensor.hpp"

#include <vector>

__C {
    llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id,
        uint8_t pin_memory) {
        if (ndim > 0 && shape == nullptr) {
            return nullptr;
        }
        try {
            std::vector<size_t> shape_vec(shape, shape + ndim);
            auto t = llaisys::Tensor::create(shape_vec, dtype, device_type, device_id, pin_memory != 0);
            if (!t) {
                return nullptr;
            }
            if (t->numel() > 0 && t->data() == nullptr) {
                return nullptr;
            }
            return new LlaisysTensor{t};
        } catch (...) {
            return nullptr;
        }
    }

    void tensorDestroy(
        llaisysTensor_t tensor) {
        if (tensor == nullptr) {
            return;
        }
        delete tensor;
    }

    void *tensorGetData(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return nullptr;
        }
        return tensor->tensor->data();
    }

    size_t tensorGetNdim(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return 0;
        }
        return tensor->tensor->ndim();
    }

    void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape) {
        if (tensor == nullptr || tensor->tensor == nullptr || shape == nullptr) {
            return;
        }
        std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
    }

    void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides) {
        if (tensor == nullptr || tensor->tensor == nullptr || strides == nullptr) {
            return;
        }
        std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
    }

    llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return LLAISYS_DTYPE_INVALID;
        }
        return tensor->tensor->dtype();
    }

    llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return LLAISYS_DEVICE_CPU;
        }
        return tensor->tensor->deviceType();
    }

    int tensorGetDeviceId(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return 0;
        }
        return tensor->tensor->deviceId();
    }

    void tensorDebug(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return;
        }
        tensor->tensor->debug();
    }

    uint8_t tensorIsContiguous(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return 0;
        }
        return uint8_t(tensor->tensor->isContiguous());
    }

    void tensorLoad(
        llaisysTensor_t tensor,
        const void *data) {
        if (tensor == nullptr || tensor->tensor == nullptr || data == nullptr) {
            return;
        }
        try {
            tensor->tensor->load(data);
        } catch (...) {
            return;
        }
    }

    llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        if (tensor == nullptr || tensor->tensor == nullptr || (ndim > 0 && shape == nullptr)) {
            return nullptr;
        }
        try {
            std::vector<size_t> shape_vec(shape, shape + ndim);
            return new LlaisysTensor{tensor->tensor->view(shape_vec)};
        } catch (...) {
            return nullptr;
        }
    }

    llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order) {
        if (tensor == nullptr || tensor->tensor == nullptr || order == nullptr) {
            return nullptr;
        }
        try {
            std::vector<size_t> order_vec(order, order + tensor->tensor->ndim());
            return new LlaisysTensor{tensor->tensor->permute(order_vec)};
        } catch (...) {
            return nullptr;
        }
    }

    llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return nullptr;
        }
        try {
            return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
        } catch (...) {
            return nullptr;
        }
    }

    llaisysTensor_t tensorContiguous(
        llaisysTensor_t tensor) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return nullptr;
        }
        try {
            return new LlaisysTensor{tensor->tensor->contiguous()};
        } catch (...) {
            return nullptr;
        }
    }

    llaisysTensor_t tensorReshape(
        llaisysTensor_t tensor,
        size_t *shape,
        size_t ndim) {
        if (tensor == nullptr || tensor->tensor == nullptr || (ndim > 0 && shape == nullptr)) {
            return nullptr;
        }
        try {
            std::vector<size_t> shape_vec(shape, shape + ndim);
            return new LlaisysTensor{tensor->tensor->reshape(shape_vec)};
        } catch (...) {
            return nullptr;
        }
    }

    llaisysTensor_t tensorTo(
        llaisysTensor_t tensor,
        llaisysDeviceType_t device_type,
        int device_id) {
        if (tensor == nullptr || tensor->tensor == nullptr) {
            return nullptr;
        }
        try {
            return new LlaisysTensor{tensor->tensor->to(device_type, device_id)};
        } catch (...) {
            return nullptr;
        }
    }
}
