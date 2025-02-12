#include <cuda/std/numeric>
#include <cuda_runtime.h>

#include "memarray.h"
#include "memarray_cuda.cuh"

namespace amso {

template <typename T, int ndim>
__global__ void
add_index_kernel(T *data, typename MemArray<T, ndim>::ArrayIndexer indexer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < indexer.size()) {
    auto indices = indexer.get_indices(idx);
    int index_sum = cuda::std::accumulate(indices.begin(), indices.end(), 0,
                                          cuda::std::plus<int>());
    data[idx] += static_cast<T>(index_sum);
  }
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <typename T, int ndim>
void add_index_tuple(MemArray<T, ndim> &arr, const int64_t stream) {
  arr.move_memory(kDLCUDA, stream, arr.get_device_id());

  // Query device properties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, arr.get_device_id());

  // Use cudaOccupancyMaxPotentialBlockSize to determine optimal block size
  int minGridSize = 0; // Minimum grid size required for full occupancy
  int blockSize = 0;   // Optimal block size
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                     add_index_kernel<T, ndim>, 0, 0);

  auto indexer = arr.get_indexer();
  // Calculate grid size based on input size and optimal block size
  int gridSize = ceil_div(indexer.size(), blockSize);

  // Ensure grid and block sizes are within device limits
  if (blockSize > deviceProp.maxThreadsPerBlock) {
    blockSize = deviceProp.maxThreadsPerBlock;
  }

  if (gridSize > deviceProp.maxGridSize[0])
    throw std::runtime_error("Error: Grid size exceeds device limits! " +
                             std::string(__FILE__) + std::to_string(__LINE__));

  // Launch the kernel with calculated configuration
  add_index_kernel<T, ndim>
      <<<gridSize, blockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
          arr.ptr(), indexer);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Kernel launch failed: " + std::string(cudaGetErrorString(err)) + " " +
        std::string(__FILE__) + std::to_string(__LINE__));
  }
}

void bind_add_index_tuple(pybind11::module &m) {
  m.def("_add_index_tuple1DInt32", &add_index_tuple<int32_t, 1>);
  m.def("_add_index_tuple2DInt32", &add_index_tuple<int32_t, 2>);
  m.def("_add_index_tuple3DInt32", &add_index_tuple<int32_t, 3>);

  m.def("_add_index_tuple1DInt64", &add_index_tuple<int64_t, 1>);
  m.def("_add_index_tuple2DInt64", &add_index_tuple<int64_t, 2>);
  m.def("_add_index_tuple3DInt64", &add_index_tuple<int64_t, 3>);

  m.def("_add_index_tuple1DFloat", &add_index_tuple<float, 1>);
  m.def("_add_index_tuple2DFloat", &add_index_tuple<float, 2>);
  m.def("_add_index_tuple3DFloat", &add_index_tuple<float, 3>);

  m.def("_add_index_tuple1DDouble", &add_index_tuple<double, 1>);
  m.def("_add_index_tuple2DDouble", &add_index_tuple<double, 2>);
  m.def("_add_index_tuple3DDouble", &add_index_tuple<double, 3>);
}

} // namespace amso
