#include "platform_status.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_amso, m) {
  m.def("print_platform_info", &printPlatformInfo,
        "Log information about the current system setup.");
}
