find_package(
  Python
  COMPONENTS Interpreter Development Development.Module
  REQUIRED)

set(PYBIND11_NEWPYTHON ON)

# Pybind11
cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 VERSION 2.10.0)

# Define the Python module
pybind11_add_module(_soma pybind_module.cpp)

# Link against your main library
target_link_libraries(_soma PUBLIC soma_core)

# Install the compiled Python module
install(TARGETS _soma LIBRARY DESTINATION soma)
install(TARGETS soma_core LIBRARY DESTINATION soma)

# Install the pure Python files
install(DIRECTORY soma/ DESTINATION soma)
