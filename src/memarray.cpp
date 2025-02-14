#include "memarray.h"

#include "memarray_pybind.h"

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thrust/async/copy.h>

namespace amso {

namespace detail {
template <typename T> struct TypeToDLPackCode {
  static constexpr DLDataTypeCode code = kDLOpaqueHandle;
  static constexpr uint8_t bits = 8 * sizeof(T);
  static constexpr uint8_t lanes = 1;
};

template <> struct TypeToDLPackCode<int32_t> {
  static constexpr DLDataTypeCode code = kDLInt;
  static constexpr uint8_t bits = 8 * sizeof(int32_t);
  static constexpr uint8_t lanes = 1;
};
template <> struct TypeToDLPackCode<int64_t> {
  static constexpr DLDataTypeCode code = kDLInt;
  static constexpr uint8_t bits = 8 * sizeof(int64_t);
  static constexpr uint8_t lanes = 1;
};
template <> struct TypeToDLPackCode<float> {
  static constexpr DLDataTypeCode code = kDLFloat;
  static constexpr uint8_t bits = 8 * sizeof(float);
  static constexpr uint8_t lanes = 1;
};
template <> struct TypeToDLPackCode<double> {
  static constexpr DLDataTypeCode code = kDLFloat;
  static constexpr uint8_t bits = 8 * sizeof(double);
  static constexpr uint8_t lanes = 1;
};

template <typename tensor_ptr_type, typename T, int ndim>
void dl_tensor_deleter(tensor_ptr_type self) {
  auto *wrapper = static_cast<MemArray<T, ndim> *>(self->manager_ctx);

  delete[] self->dl_tensor.shape;
  delete[] self->dl_tensor.strides;
  // Invalidate tensor
  self->dl_tensor.data = nullptr;
  self->dl_tensor.ndim = 0;
  self->dl_tensor.shape = nullptr;
  self->dl_tensor.strides = nullptr;
  self->dl_tensor.byte_offset = 0;
}
template <typename tensor_ptr_type> void dl_capsule_deleter(PyObject *capsule) {
  void *raw_ptr = nullptr;
  // Can be original name if unused "dltensor"
  if (strcmp("dltensor", PyCapsule_GetName(capsule)) == 0)
    raw_ptr = PyCapsule_GetPointer(capsule, "dltensor");
  else // "used_dltensor if capsule is consumed
    raw_ptr = PyCapsule_GetPointer(capsule, "used_dltensor");

  if (raw_ptr) // Unknown capsule or already freed capsule
  {
    tensor_ptr_type tensor_ptr = static_cast<tensor_ptr_type>(raw_ptr);
    if (tensor_ptr->deleter) // Execute custom deleter, here delete[] shape.
      tensor_ptr->deleter(tensor_ptr);
  }
}
}; // namespace detail

template <typename T, int ndim>
MemArray<T, ndim>::MemArray(const std::array<int, ndim> &shape, int device_id)
    : _shape({0}), _on_device(false), _device_id(device_id), _lock_id(-1) {

  for (auto shape_element : shape) {
    if (shape_element <= 0) {
      std::string msg = "Invalid shape argument, all shapes have to be "
                        "bigger then 0 but got < ";
      for (auto invalid_shape : shape)
        msg += std::to_string(invalid_shape) + " ";
      msg += "> where " + std::to_string(shape_element) + " is invalid";
      throw std::invalid_argument(msg);
    }
  }
  int64_t size = 1;
  for (auto shape_element : shape)
    size *= shape_element;

  // Ensure proper device
  cudaSetDevice(device_id);

  // Init internals
  _host_vec = thrust::host_vector<T>(size);
  _device_vec = thrust::device_vector<T>(size);

  _shape = shape;
}

template <typename T, int ndim>
cudaStream_t
MemArray<T, ndim>::convert_python_int_to_stream(const int64_t stream_id) {
  cudaStream_t stream = cudaStreamLegacy;
  switch (stream_id) {
  case 0:
    stream = cudaStreamLegacy;
    break;
  case 1:
    stream = cudaStreamPerThread;
    break;
  default:
    stream = reinterpret_cast<cudaStream_t>(
        stream_id); // This reinterpret_cast is a little suspicious, however
    // PyTorch does it that way. So it should be fine.
  };
  return stream;
}

template <typename T, int ndim>
void MemArray<T, ndim>::move_memory(const DLDeviceType requested_device_type,
                                    const int64_t requested_cuda_stream_id,
                                    const int requested_device_id,
                                    const int64_t requested_lock_id,
                                    const bool async) {
  // Sanity checks
  throw_invalid_lock_access(requested_lock_id);

  if (requested_device_id > 0 and get_device_id() != requested_device_id)
    throw std::runtime_error("Requested to move memory to different device (" +
                             std::to_string(requested_device_id) +
                             ", not supported. Current device " +
                             std::to_string(get_device_id()));

  const cudaStream_t stream =
      convert_python_int_to_stream(requested_cuda_stream_id);

  if (not async) {
    cudaStreamSynchronize(stream);
  }

  thrust::device_event event;
  // Memory movement
  switch (requested_device_type) {
  case kDLCUDA:
    if (get_on_device()) // EARLY exit
      return;

    event = thrust::async::copy(thrust::host, thrust::cuda::par.on(stream),
                                _host_vec.begin(), _host_vec.end(),
                                _device_vec.begin());
    _on_device = true;
    break;
  case kDLCPU:
    if (not get_on_device()) // EARLY EXIT
      return;
    event = thrust::async::copy(thrust::cuda::par.on(stream), thrust::host,
                                _device_vec.begin(), _device_vec.end(),
                                _host_vec.begin());

    _on_device = false;
    break;
  default:
    throw std::runtime_error("Unsupported memory move requested " +
                             std::to_string(requested_device_type) + ".");
  }

  if (not async) {
    event.wait();
    cudaStreamSynchronize(stream);
  }
}

template <typename T, int ndim>
pybind11::capsule
MemArray<T, ndim>::get_dlpack_tensor(const bool versioned,
                                     const int64_t requested_lock_id) {
  // Access to internal data
  T *data_ptr = this->ptr(requested_lock_id);

  DLTensor dl_tensor;
  dl_tensor.data = data_ptr;
  dl_tensor.shape = new int64_t[ndim]; // Throws std::bad_alloc on failure
  std::copy(get_shape().begin(), get_shape().end(),
            dl_tensor.shape); // Init with correct shape

  auto device_pair = get_dlpack_device();
  dl_tensor.device.device_type = device_pair.first;
  dl_tensor.device.device_id = device_pair.second;
  dl_tensor.ndim = ndim;
  dl_tensor.dtype = DLDataType{detail::TypeToDLPackCode<T>::code,
                               detail::TypeToDLPackCode<T>::bits,
                               detail::TypeToDLPackCode<T>::lanes};
  dl_tensor.strides = nullptr;
  dl_tensor.byte_offset = 0;

  if (versioned) {
    auto versioned_tensor = std::make_unique<DLManagedTensorVersioned>();

    versioned_tensor->version =
        DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    versioned_tensor->manager_ctx = this;
    versioned_tensor->deleter =
        &detail::dl_tensor_deleter<DLManagedTensorVersioned *, T, ndim>;
    versioned_tensor->flags = 0;
    versioned_tensor->dl_tensor = dl_tensor;

    // Release unique pointer to capsule, as we transfer ownership to the
    // python capsule.
    return pybind11::capsule(
        versioned_tensor.release(), "dltensor",
        &detail::dl_capsule_deleter<DLManagedTensorVersioned *>);
  } else { // Versioned DLtensor not supported
    auto tensor = std::make_unique<DLManagedTensor>();
    tensor->dl_tensor = dl_tensor;
    tensor->manager_ctx = this;
    tensor->deleter = &detail::dl_tensor_deleter<DLManagedTensor *, T, ndim>;

    // Release unique pointer to capsule, as we transfer ownership to the
    // python capsule.
    return pybind11::capsule(tensor.release(), "dltensor",
                             &detail::dl_capsule_deleter<DLManagedTensor *>);
  }
}

template <typename T, int ndim>
std::pair<DLDeviceType, int32_t> MemArray<T, ndim>::get_dlpack_device() const {
  if (get_on_device())
    return std::make_pair(DLDeviceType::kDLCUDA, get_device_id());
  return std::make_pair(DLDeviceType::kDLCPU, 0);
}

template <typename T, int ndim>
void MemArray<T, ndim>::read_numpy_array(
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
        np_array) {
  move_memory(kDLCPU, 0, get_device_id(), -1, true);
  pybind11::buffer_info buf = np_array.request();

  if (buf.ndim != ndim)
    throw std::invalid_argument("Input array has " + std::to_string(buf.ndim) +
                                " dimensions, but target memarray has " +
                                std::to_string(ndim) + " dimensions.");

  for (auto i = 0; i < ndim; ++i) {
    if (buf.shape[i] != get_shape()[i]) {
      throw std::invalid_argument(
          "Input array has incompatible shapes at pos " + std::to_string(i) +
          " input has " + std::to_string(buf.shape[i]) + " MemArray has " +
          std::to_string(get_shape()[i]) + ".");
    }
  }

  thrust::copy_n(thrust::host, static_cast<T *>(buf.ptr), get_size(),
                 _host_vec.begin());
}

template <typename T, int ndim>
const T MemArray<T, ndim>::operator()(const std::array<int, ndim> &indeces,
                                      const int64_t requested_lock_id) const {
  throw_invalid_lock_access(requested_lock_id);

  for (int i = 0; i < ndim; ++i)
    if (indeces[i] >= get_shape()[i])
      throw std::runtime_error("Invalid index for shape access");
  auto indexer = ArrayIndexer(get_shape());
  if (get_on_device())
    return _device_vec[indexer(indeces)];
  return _host_vec[indexer(indeces)];
}

template <typename T, int ndim>
void MemArray<T, ndim>::write(const std::array<int, ndim> &indeces,
                              const T &value, const int64_t requested_lock_id) {
  throw_invalid_lock_access(requested_lock_id);
  for (int i = 0; i < ndim; ++i)
    if (indeces[i] >= get_shape()[i])
      throw std::runtime_error("Invalid index for write access");
  auto indexer = ArrayIndexer(get_shape());
  if (get_on_device())
    _device_vec[indexer(indeces)] = value;
  else
    _host_vec[indexer(indeces)] = value;
}

template <typename MemArrayType>
void template_bind_mem_array(pybind11::module &m, std::string python_name) {
  pybind11::class_<MemArrayType>(m, python_name.c_str())
      .def(pybind11::init<const std::array<int, MemArrayType::NDIM> &, int>())

      .def("_enter", &MemArrayType::enter)
      .def("_exit", &MemArrayType::exit)
      .def("_move_memory", &MemArrayType::move_memory)
      .def("_dlpack", &MemArrayType::get_dlpack_tensor)
      .def("_read_numpy_array", &MemArrayType::read_numpy_array)
      .def("_dlpack_device", &MemArrayType::get_dlpack_device);
}

// Explicit template instantiations
template class MemArray<int32_t, 1>;
using MemArray1DInt32 = MemArray<int32_t, 1>;
template class MemArray<int32_t, 2>;
using MemArray2DInt32 = MemArray<int32_t, 2>;
template class MemArray<int32_t, 3>;
using MemArray3DInt32 = MemArray<int32_t, 3>;

template class MemArray<int64_t, 1>;
using MemArray1DInt64 = MemArray<int64_t, 1>;
template class MemArray<int64_t, 2>;
using MemArray2DInt64 = MemArray<int64_t, 2>;
template class MemArray<int64_t, 3>;
using MemArray3DInt64 = MemArray<int64_t, 3>;

template class MemArray<float, 1>;
using MemArray1DFloat = MemArray<float, 1>;
template class MemArray<float, 2>;
using MemArray2DFloat = MemArray<float, 2>;
template class MemArray<float, 3>;
using MemArray3DFloat = MemArray<float, 3>;

template class MemArray<double, 1>;
using MemArray1DDouble = MemArray<double, 1>;
template class MemArray<double, 2>;
using MemArray2DDouble = MemArray<double, 2>;
template class MemArray<double, 3>;
using MemArray3DDouble = MemArray<double, 3>;

void bind_mem_array(pybind11::module &m) {
  template_bind_mem_array<MemArray1DInt32>(m, "MemArray1DInt32");
  template_bind_mem_array<MemArray2DInt32>(m, "MemArray2DInt32");
  template_bind_mem_array<MemArray3DInt32>(m, "MemArray3DInt32");

  template_bind_mem_array<MemArray1DInt64>(m, "MemArray1DInt64");
  template_bind_mem_array<MemArray2DInt64>(m, "MemArray2DInt64");
  template_bind_mem_array<MemArray3DInt64>(m, "MemArray3DInt64");

  template_bind_mem_array<MemArray1DFloat>(m, "MemArray1DFloat");
  template_bind_mem_array<MemArray2DFloat>(m, "MemArray2DFloat");
  template_bind_mem_array<MemArray3DFloat>(m, "MemArray3DFloat");

  template_bind_mem_array<MemArray1DDouble>(m, "MemArray1DDouble");
  template_bind_mem_array<MemArray2DDouble>(m, "MemArray2DDouble");
  template_bind_mem_array<MemArray3DDouble>(m, "MemArray3DDouble");
}

} // namespace amso
