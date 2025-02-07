#include "memarray.h"

namespace py = pybind11;

void pybind_memarray(py::module_ &m) {
  amso::bind_mem_array<int32_t, 1>(m, "MemArray1DInt32");
  amso::bind_mem_array<int32_t, 2>(m, "MemArray2DInt32");
  amso::bind_mem_array<int32_t, 3>(m, "MemArray3DInt32");

  amso::bind_mem_array<int64_t, 1>(m, "MemArray1DInt64");
  amso::bind_mem_array<int64_t, 2>(m, "MemArray2DInt64");
  amso::bind_mem_array<int64_t, 3>(m, "MemArray3DInt64");

  amso::bind_mem_array<float, 1>(m, "MemArray1DFloat");
  amso::bind_mem_array<float, 2>(m, "MemArray2DFloat");
  amso::bind_mem_array<float, 3>(m, "MemArray3DFloat");

  amso::bind_mem_array<double, 1>(m, "MemArray1DDouble");
  amso::bind_mem_array<double, 2>(m, "MemArray2DDouble");
  amso::bind_mem_array<double, 3>(m, "MemArray3DDouble");
}
