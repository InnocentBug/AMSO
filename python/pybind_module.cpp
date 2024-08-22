#include "platform_status.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_soma, m) {
  m.def("print_platform_info", &printPlatformInfo,
        "Log information about the current system setup.");
}
