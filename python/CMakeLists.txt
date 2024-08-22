find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Define the Python module
pybind11_add_module(_soma pybind_module.cpp)

# Link against your main library
target_link_libraries(_soma PRIVATE SOMA2)
