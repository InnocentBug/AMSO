#include <array>

#include "memarray_cuda.cuh"
#include <dlpack/dlpack.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#pragma once

namespace amso {

template <typename T, int ndim> class MemArray {
private:
  thrust::host_vector<T> _host_vec;
  thrust::device_vector<T> _device_vec;

  std::array<int64_t, ndim> _shape;
  int64_t _size;
  bool _on_device;
  int _device_id;
  int64_t _lock_id; // Context manager lock, ID. Negative means unlocked.
                    // Positive, locked to that context manager

private:
  const std::array<int64_t, ndim> &get_shape() const { return _shape; }
  int64_t get_size() const { return _size; }
  bool get_on_device() const { return _on_device; }
  int get_device_id() const { return _device_id; }
  int64_t get_lock_id() const { return _lock_id; }

  cudaStream_t convert_python_int_to_stream(const int64_t stream_id);

  void to_device(const int64_t requested_cuda_stream_id,
                 const int requested_device_id,
                 const int64_t requested_lock_id = -1);

  void to_host(const int64_t requested_cuda_stream_id,
               const int requested_device_id,
               const int64_t requested_lock_id = -1);

public:
  MemArray(const std::array<int64_t, ndim> &shape, int device_id);

  void move_memory(const DLDeviceType requested_device_type,
                   const int64_t requested_cuda_stream_id,
                   const int requested_device_id,
                   const int64_t requested_lock_id = -1);

  pybind11::capsule get_dlpack_tensor(const bool versioned,
                                      const int64_t requested_lock_id);

  std::pair<DLDeviceType, int32_t> get_dlpack_device() const;

  // Context manager methods
  void enter(const int64_t lock_id) { _lock_id = lock_id; }
  void exit() { _lock_id = -1; }

  void read_numpy_array(pybind11::array_t<T, pybind11::array::c_style |
                                                 pybind11::array::forcecast>
                            np_array);
};

template <typename T, int ndim>
void template_bind_mem_array(pybind11::module &m, std::string python_name);

} // namespace amso
