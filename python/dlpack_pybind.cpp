#include "dlpack_pybind.h"
#include <dlpack/dlpack.h>

namespace py = pybind11;

void pybind_dlpack(py::module_ &m) {
  py::module_ dlpack = m.def_submodule("dlpack", "DLpack python binding.");
  py::enum_<DLDeviceType>(dlpack, "DLDeviceType")
      .value("kDLCPU", DLDeviceType::kDLCPU)
      .value("kDLCUDA", DLDeviceType::kDLCUDA)
      .value("kDLCUDAHost", DLDeviceType::kDLCUDAHost)
      .value("kDLOpenCL", DLDeviceType::kDLOpenCL)
      .value("kDLVulkan", DLDeviceType::kDLVulkan)
      .value("kDLMetal", DLDeviceType::kDLMetal)
      .value("kDLVPI", DLDeviceType::kDLVPI)
      .value("kDLROCM", DLDeviceType::kDLROCM)
      .value("kDLROCMHost", DLDeviceType::kDLROCMHost)
      .value("kDLExtDev", DLDeviceType::kDLExtDev)
      .value("kDLCUDAManaged", DLDeviceType::kDLCUDAManaged)
      .value("kDLOneAPI", DLDeviceType::kDLOneAPI)
      .value("kDLWebGPU", DLDeviceType::kDLWebGPU)
      // .value("kDLHexagon", kDLHexagon::kDLHexagon)
      .value("kDLMAIA", kDLMAIA)
      .export_values(); // DLPack is C, so we don't have strongly tryped enums
  dlpack.attr("DLPACK_MAJOR_VERSION") = DLPACK_MAJOR_VERSION;
  dlpack.attr("DLPACK_MINOR_VERSION") = DLPACK_MINOR_VERSION;
}
