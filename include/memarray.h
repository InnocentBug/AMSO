#ifndef MEMARRAY_H
#define MEMARRAY_H

#include <cuda/std/array>
#include <cuda/std/numeric>
#include <dlpack/dlpack.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace amso {

namespace detail {
template <typename T, int ndim>
cuda::std::array<T, ndim>
make_cuda_std_array(const std::array<T, ndim> &std_arr) {
  cuda::std::array<T, ndim> cuda_std_arr;
  for (int i = 0; i < ndim; ++i)
    cuda_std_arr[i] = std_arr[i];
  return cuda_std_arr;
}
template <typename T, int ndim>
std::array<T, ndim>
make_std_cuda_array(const cuda::std::array<T, ndim> &cuda_std_arr) {
  std::array<T, ndim> std_arr;
  for (int i = 0; i < ndim; ++i)
    std_arr[i] = cuda_std_arr[i];
  return std_arr;
}
} // namespace detail

template <typename T, int ndim> class MemArray;

template <typename T, int ndim>
void swap(MemArray<T, ndim> &lhs, MemArray<T, ndim> &rhs) noexcept;

template <typename T, int ndim> class MemArray {
private:
  thrust::host_vector<T> _host_vec;
  thrust::device_vector<T> _device_vec;

  std::array<int, ndim> _shape;
  int64_t _size;
  bool _on_device;
  int _device_id;
  int64_t _lock_id; // Context manager lock, ID. Negative means unlocked.
                    // Positive, locked to that context manager

private:
  cudaStream_t convert_python_int_to_stream(const int64_t stream_id);

public:
  static constexpr int NDIM = ndim;
  class ArrayIndexer {
  private:
    const cuda::std::array<int, ndim> _shape;

  public:
    inline ArrayIndexer(const std::array<int, ndim> &shape)
        : _shape(detail::make_cuda_std_array<int, ndim>(shape)) {
      for (auto s : shape)
        assert(s > 0);
    }

    __host__ __device__ inline int
    operator()(const cuda::std::array<int, ndim> &indeces) const {
      int index = 0;
      int stride = 1;
      for (int i = ndim - 1; i >= 0; --i) {
        assert(indeces[i] < _shape[i]);
        index += indeces[i] * stride;
        stride *= _shape[i];
      }
      return index;
    }
    __host__ inline int operator()(const std::array<int, ndim> &indeces) const {
      return this->operator()(detail::make_cuda_std_array<int, ndim>(indeces));
    }

    __host__ __device__ inline cuda::std::array<int, ndim>
    get_indices(int idx) const {
      cuda::std::array<int, ndim> indices;
      for (int i = ndim - 1; i >= 0; --i) {
        indices[i] = idx % _shape[i];
        idx /= _shape[i];
      }
      return indices;
    }
    __host__ __device__ inline int size() const {
      return cuda::std::accumulate(_shape.begin(), _shape.end(), 1,
                                   cuda::std::multiplies<int>());
    }
  };

public:
  MemArray(const std::array<int, ndim> &shape,
           int device_id);                 // Regular constructor for new memory
  ~MemArray(){};                           // Destructor
  MemArray(const MemArray<T, ndim> &other) // Copy constructor (expensive)
      : _host_vec(other._host_vec), _device_vec(other._device_vec),
        _shape(other._shape), _size(other._size), _on_device(other._on_device),
        _device_id(other._device_id), _lock_id(other._lock_id) {}

  void swap(MemArray<T, ndim> &other) noexcept {
    using std::swap;
    swap(_host_vec, other._host_vec);
    swap(_device_vec, other._device_vec);
    swap(_shape, other._shape);
    swap(_size, other._size);
    swap(_on_device, other._on_device);
    swap(_device_id, other._device_id);
    swap(_lock_id, other._lock_id);
  } // Efficient swap (cheap)
  template <typename U, int N>
  friend void swap(MemArray<U, N> &lhs, MemArray<U, N> &rhs) noexcept;

  MemArray(MemArray<T, ndim> &&other) noexcept { // Move constructor (cheap)
    this->swap(other);
  }

  MemArray<T, ndim> &
  operator=(const MemArray<T, ndim> &other) { // Copy Assignment operator
    MemArray<T, ndim> temp(other);
    this->swap(temp);
    return *this;
  }

  MemArray<T, ndim> &
  operator=(MemArray<T, ndim> &&other) noexcept { // Move Assignment operator
    this->swap(other);
    return *this;
  }

  int get_device_id() const { return _device_id; }
  const std::array<int, ndim> &get_shape() const { return _shape; }
  int64_t get_size() const { return _size; }
  bool get_on_device() const { return _on_device; }
  int64_t get_lock_id() const { return _lock_id; }

  void move_memory(const DLDeviceType requested_device_type,
                   const int64_t requested_cuda_stream_id,
                   const int requested_device_id,
                   const int64_t requested_lock_id = -1,
                   const bool async = true);

  pybind11::capsule get_dlpack_tensor(const bool versioned,
                                      const int64_t requested_lock_id);

  std::pair<DLDeviceType, int32_t> get_dlpack_device() const;

  // Context manager methods
  void enter(const int64_t lock_id) { _lock_id = lock_id; }
  void exit() { _lock_id = -1; }

  void read_numpy_array(pybind11::array_t<T, pybind11::array::c_style |
                                                 pybind11::array::forcecast>
                            np_array);

  // encaspulated access access, not zero-overhead
  const T operator()(const std::array<int, ndim> &indeces) const;
  const T operator()(const cuda::std::array<int, ndim> &indeces) const {
    return this->operator()(detail::make_std_cuda_array<int, ndim>(indeces));
  }

  void write(const std::array<int, ndim> &indeces, const T &value);
  void write(const cuda::std::array<int, ndim> &indeces, const T &value) {
    return this->write(detail::make_std_cuda_array<int, ndim>(indeces), value);
  }

  // Zero over-head pointer access
  // Taking the sharp knives out of the drawer: don't cut yourself
  T *ptr() {
    if (_on_device)
      return thrust::raw_pointer_cast(_device_vec.data());
    return _host_vec.data();
  }
  const T *ptr() const { return const_cast<const T *>(this->ptr()); }

  ArrayIndexer get_indexer() const { return ArrayIndexer(get_shape()); }
};
template <typename T, int ndim>
void swap(MemArray<T, ndim> &lhs, MemArray<T, ndim> &rhs) noexcept {
  lhs.swap(rhs);
}

} // namespace amso

#endif // MEMARRAY_H
