#include <array>

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#pragma once

namespace amso {

template <typename T, int ndim> class MemArray; // Forward Declaration

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

template <typename T, int ndim> class MemArray {
private:
  thrust::host_vector<T> host_vec; // Pinned memory
  thrust::device_vector<T> device_vec;

  std::array<int64_t, ndim> shape;
  int64_t size;

  bool on_device;
  int device_id;

  int lock_count; // Context manager locks

  // Keeping track of dlpack capsule to see if they have been released
  std::map<int64_t *, bool> dlpack_capsule_lock;

private:
  bool all_dlpack_capsules_unlocked() {
    for (auto iter : dlpack_capsule_lock) {
      if (iter.second)
        return false;
      std::cout << "test: " << iter.first << " " << iter.second << std::endl;
    }
    return true;
  }

public:
  MemArray(const std::array<int64_t, ndim> &shape_, int device_id_)
      : shape({0}), size(0), on_device(false), device_id(device_id_),
        lock_count(0) {

    for (auto shape_element : shape_) {
      if (shape_element <= 0) {
        std::string msg = "Invalid shape argument, all shapes have to be "
                          "bigger then 0 but got < ";
        for (auto invalid_shape : shape)
          msg += std::to_string(invalid_shape) + " ";
        msg += "> where " + std::to_string(shape_element) + " is invalid";
        throw std::invalid_argument(msg);
      }

      size = 1;
      for (auto shape_element : shape_)
        size *= shape_element;

      host_vec = thrust::host_vector<T>(size);
      shape = shape_;
    }
  }

  pybind11::capsule get_dlpack_tensor(bool versioned) {
    if (lock_count == 0)
      throw std::runtime_error("Accessing memory without active context.");

    T *data_ptr = on_device ? thrust::raw_pointer_cast(device_vec.data())
                            : host_vec.data();

    DLTensor dl_tensor;
    dl_tensor.data = data_ptr;
    dl_tensor.shape = new int64_t[ndim]; // Throws std::bad_alloc on failure
    std::copy(shape.begin(), shape.end(),
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

  std::pair<DLDeviceType, int32_t> get_dlpack_device() {
    if (on_device)
      return std::make_pair(DLDeviceType::kDLCUDA, device_id);
    return std::make_pair(DLDeviceType::kDLCPU, 0);
  }

  // Context manager methods
  void enter() { lock_count++; }
  void exit() { lock_count--; }

  void to_device() {
    if (on_device)
      return;

    if (lock_count > 0)
      throw std::runtime_error("Cannot move memory while locked to device");

    cudaSetDevice(device_id);
    // thrust::copy_n(thrust::device, host_vec.begin(), size,
    // device_vec.begin());
    device_vec = host_vec;
    on_device = true;
  }

  void to_host() {
    if (not on_device)
      return;

    if (lock_count > 0)
      throw std::runtime_error("Cannot move memory while locked to host");

    cudaSetDevice(device_id);
    // thrust::copy_n(thrust::device, device_vec.begin(), size,
    // host_vec.begin());
    host_vec = device_vec;
    on_device = false;
  }

  int get_lock_count() { return lock_count; }

  void read_numpy_array(pybind11::array_t<T, pybind11::array::c_style |
                                                 pybind11::array::forcecast>
                            np_array) {
    to_host();
    pybind11::buffer_info buf = np_array.request();

    if (buf.ndim != ndim)
      throw std::invalid_argument("Input array has " +
                                  std::to_string(buf.ndim) +
                                  " dimensions, but target memarray has " +
                                  std::to_string(ndim) + " dimensions.");

    for (auto i = 0; i < ndim; ++i) {
      if (buf.shape[i] != shape[i]) {
        throw std::invalid_argument(
            "Input array has incompatible shapes at pos " + std::to_string(i) +
            " input has " + std::to_string(buf.shape[i]) + " MemArray has " +
            std::to_string(shape[i]) + ".");
      }
    }

    thrust::copy_n(thrust::host, static_cast<T *>(buf.ptr), size,
                   host_vec.begin());
  }
};

template <typename T, int ndim>
void bind_mem_array(pybind11::module &m, std::string python_name) {
  pybind11::class_<MemArray<T, ndim>>(m, python_name.c_str())
      .def(pybind11::init<const std::array<int64_t, ndim> &, int>())

      .def("_enter", &MemArray<T, ndim>::enter)
      .def("_exit", &MemArray<T, ndim>::exit)
      .def("_to_device", &MemArray<T, ndim>::to_device)
      .def("_to_host", &MemArray<T, ndim>::to_host)
      .def("_dlpack", &MemArray<T, ndim>::get_dlpack_tensor)
      .def("_read_numpy_array", &MemArray<T, ndim>::read_numpy_array)
      .def("_dlpack_device", &MemArray<T, ndim>::get_dlpack_device);
}

} // namespace amso
