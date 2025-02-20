#include <pybind11/pybind11.h>

#include "dlpack_pybind.h"
#include "memarray_cuda.cuh"
#include "memarray_pybind.h"

namespace py = pybind11;

PYBIND11_MODULE(_amso, m) {
  amso::pybind_dlpack(m);
  amso::bind_mem_array(m);
  amso::bind_add_index_tuple(m);
}
