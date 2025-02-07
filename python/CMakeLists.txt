# Pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(_amso_python_sources memarray_pybind.cpp dlpack_pybind.cpp)
add_library(amso_python ${_amso_python_sources})
target_link_libraries(amso_python PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(amso_python PRIVATE pybind11::pybind11)
set_target_properties(amso_python PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(amso_python PUBLIC ${dlpack_SOURCE_DIR}/include/)
target_include_directories(amso_python
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Define the Python module
pybind11_add_module(_amso module_amso.cpp)

# Link against your main library
target_link_libraries(_amso PUBLIC amso_core amso_python)
target_link_libraries(_amso PRIVATE pybind11::pybind11)
set_target_properties(_amso PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Install the compiled Python module
install(TARGETS _amso LIBRARY DESTINATION amso)
