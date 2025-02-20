#ifndef MEMARRAY_CUDA_CUH
#define MEMARRAY_CUDA_CUH

#include <pybind11/pybind11.h>
#include "memarray.h"

namespace amso{
  
  template<typename T, int ndim>
  void add_index_tuple(MemArray<T,ndim>&arr, const int64_t stream=0);
  void bind_add_index_tuple(pybind11::module &m);
  
} //namespace amso
#endif // MEMARRAY_CUDA_CUH
