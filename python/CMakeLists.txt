# Define the Python module
pybind11_add_module(pybind_module pybind_module.cpp)

# Link against your main library
target_link_libraries(pybind_module PRIVATE SOMA2)
