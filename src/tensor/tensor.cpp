#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

#ifdef ENABLE_NVIDIA_API
namespace llaisys::device::nvidia {
void contiguous_strided_copy(void *dst,
                             const void *src,
                             const size_t *shape,
                             const ptrdiff_t *strides,
                             size_t ndim,
                             size_t numel,
                             size_t elem_size,
                             llaisysStream_t stream);
} // namespace llaisys::device::nvidia
#endif

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    ptrdiff_t expected = 1;
    for (size_t i = 0; i < _meta.shape.size(); ++i) {
        size_t dim_index = this->ndim() - 1 - i;
        if (_meta.strides[dim_index] != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(_meta.shape[dim_index]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t n_dim = this->ndim();
    CHECK_ARGUMENT(order.size() == n_dim, "permute: order size mismatch");
    std::vector<int> seen(n_dim, 0);
    for (size_t i = 0; i < n_dim; ++i) {
        CHECK_ARGUMENT(order[i] < n_dim, "permute: order index out of range");
        CHECK_ARGUMENT(seen[order[i]] == 0, "permute: repeated index");
        seen[order[i]] = 1;
    }
    TensorMeta meta = _meta;
    for (size_t i = 0; i < n_dim; ++i) {
        meta.shape[i] = _meta.shape[order[i]];
        meta.strides[i] = _meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t n_e = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        n_e *= shape[i];
    }
    CHECK_ARGUMENT(n_e == this->numel(), "view: shape size mismatch");
    CHECK_ARGUMENT(this->isContiguous(), "view: only support contiguous stride");
    std::vector<ptrdiff_t> new_stride(shape.size());
    size_t tmp_stride = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t dim_index = shape.size() - 1 - i;
        new_stride[dim_index] = tmp_stride;
        tmp_stride *= shape[dim_index];  
    }
    TensorMeta meta{_meta.dtype, shape, new_stride};
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "slice: dim out of range");
    CHECK_ARGUMENT(start <= end, "slice: start > end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice: end out of range");  
    TensorMeta meta  = _meta;
    meta.shape[dim] = end - start;
    size_t offset = _offset + static_cast<size_t>(_meta.strides[dim] * start * this->elementSize());
    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto api = core::context().runtime().api();
    size_t bytes = this->numel() * this->elementSize(); 
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        api->memcpy_sync(this->data(), src_, bytes, LLAISYS_MEMCPY_H2H);
    } else {
        api->memcpy_sync(this->data(), src_, bytes, LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    auto out = create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());
    const size_t elem_size = this->elementSize();
    const size_t ndim_ = this->ndim();
    const auto &shape_ = this->shape();
    const auto &strides_ = this->strides();
    const size_t total = this->numel();
    std::vector<size_t> idx(ndim_, 0);
    std::vector<std::byte> host_out(total * elem_size);

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        const std::byte *src = this->data();
        for (size_t linear = 0; linear < total; ++linear) {
            ptrdiff_t src_elem_offset = 0;
            for (size_t d = 0; d < ndim_; ++d) {
                src_elem_offset += static_cast<ptrdiff_t>(idx[d]) * strides_[d];
            }
            std::memcpy(host_out.data() + linear * elem_size, src + static_cast<size_t>(src_elem_offset) * elem_size, elem_size);

            for (size_t r = ndim_; r > 0; --r) {
                const size_t d = r - 1;
                idx[d] += 1;
                if (idx[d] < shape_[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
        std::memcpy(out->data(), host_out.data(), host_out.size());
        return out;
    }

    // Fast path for NVIDIA device tensors: do strided gather directly on device.
#ifdef ENABLE_NVIDIA_API
    if (this->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        core::context().setDevice(this->deviceType(), this->deviceId());
        device::nvidia::contiguous_strided_copy(out->data(),
                                                this->data(),
                                                shape_.data(),
                                                strides_.data(),
                                                ndim_,
                                                total,
                                                elem_size,
                                                core::context().runtime().stream());
        core::context().runtime().synchronize();
        return out;
    }
#endif

    // Conservative device fallback: stage visible storage tail to host, gather on host, copy back.
    const size_t staged_bytes = _storage->size() - _offset;
    std::vector<std::byte> host_stage(staged_bytes);
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->memcpy_sync(host_stage.data(), this->data(), staged_bytes, LLAISYS_MEMCPY_D2H);
    const std::byte *src_stage = host_stage.data();

    for (size_t linear = 0; linear < total; ++linear) {
        ptrdiff_t src_elem_offset = 0;
        for (size_t d = 0; d < ndim_; ++d) {
            src_elem_offset += static_cast<ptrdiff_t>(idx[d]) * strides_[d];
        }
        std::memcpy(host_out.data() + linear * elem_size, src_stage + static_cast<size_t>(src_elem_offset) * elem_size, elem_size);

        for (size_t r = ndim_; r > 0; --r) {
            const size_t d = r - 1;
            idx[d] += 1;
            if (idx[d] < shape_[d]) {
                break;
            }
            idx[d] = 0;
        }
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    core::context().runtime().api()->memcpy_sync(out->data(), host_out.data(), host_out.size(), LLAISYS_MEMCPY_H2D);
    return out;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t n_e = 1;
    for (size_t d : shape) {
        n_e *= d;
    }
    CHECK_ARGUMENT(n_e == this->numel(), "reshape: shape size mismatch");
    if (this->isContiguous()) {
        return this->view(shape);
    }
    return this->contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    const int dst_device = (device >= 0) ? device : ((device_type == this->deviceType()) ? this->deviceId() : 0);
    if (device_type == this->deviceType() && dst_device == this->deviceId()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    auto src_ctg = this->isContiguous() ? std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset)) : this->contiguous();
    auto dst = create(src_ctg->shape(), src_ctg->dtype(), device_type, dst_device);
    const size_t bytes = src_ctg->numel() * src_ctg->elementSize();
    if (bytes == 0) {
        return dst;
    }

    llaisysMemcpyKind_t kind = LLAISYS_MEMCPY_H2H;
    if (src_ctg->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
        core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
    } else if (src_ctg->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
        core::context().setDevice(device_type, dst_device);
    } else if (src_ctg->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
        core::context().setDevice(src_ctg->deviceType(), src_ctg->deviceId());
    } else {
        CHECK_ARGUMENT(src_ctg->deviceType() == device_type, "to: cross-backend device copy is not supported");
        kind = LLAISYS_MEMCPY_D2D;
        core::context().setDevice(src_ctg->deviceType(), src_ctg->deviceId());
    }

    core::context().runtime().api()->memcpy_sync(dst->data(), src_ctg->data(), bytes, kind);
    return dst;
}

} // namespace llaisys
