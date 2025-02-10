# Add your source files
set(_amso_cpu_sources dlpack_pybind.cpp memarray.cpp)
set_source_files_properties(${_amso_cpu_sources} PROPERTIES LANGUAGE CUDA)

# Define the target first
add_library(amso_core ${_amso_cpu_sources})

if(ENABLE_CUDA)
  set(_amso_cuda_sources memarray_cuda.cu)

  target_link_libraries(amso_core PRIVATE CUDA::cudart CUDA::cuda_driver)
  # CUDA specific settings
  set_target_properties(amso_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(amso_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  # Set CUDA language for the source files
  set_source_files_properties(${_amso_cuda_sources} PROPERTIES LANGUAGE CUDA)

  # Add the CUDA sources to the target sources
  set(_amso_sources ${_amso_cpu_sources} ${_amso_cuda_sources})
else()
  set(_amso_sources ${_amso_cpu_sources})
endif()

target_link_libraries(amso_core PRIVATE Python::Python)

# Update target sources
target_sources(amso_core PRIVATE ${_amso_sources})

# Set properties
set_target_properties(amso_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Include directories
target_include_directories(amso_core
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# DLPACK
cpmaddpackage(NAME dlpack GITHUB_REPOSITORY dmlc/dlpack GIT_TAG main)
target_include_directories(amso_core PUBLIC ${dlpack_SOURCE_DIR}/include/)
target_include_directories(amso_core PRIVATE ${pybind11_SOURCE_DIR}/include/)
