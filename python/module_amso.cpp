#include <pybind11/pybind11.h>

#include "dlpack_pybind.h"
#include "memarray_pybind.h"

namespace py = pybind11;

PYBIND11_MODULE(_amso, m) {
  pybind_memarray(m);
  pybind_dlpack(m);
}
