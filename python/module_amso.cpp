#include <pybind11/pybind11.h>

#include "memarray.h"
#include "platform_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_amso, m) {
  m.def("print_platform_info", &printPlatformInfo,
        "Log information about the current system setup.");

#include "dlpack_pybind.incl"
#include "memarray_pybind.incl"
}
