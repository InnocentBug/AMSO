find_package(
  Python
  COMPONENTS Interpreter Development Development.Module
  REQUIRED)

set(PYBIND11_NEWPYTHON ON)

# Pybind11
cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 VERSION 2.10.0)

# Define the Python module
pybind11_add_module(_amso pybind_module.cpp)

# Link against your main library
target_link_libraries(_amso PUBLIC amso_core pybind11::module Python::Python)

# Install the compiled Python module
install(TARGETS _amso LIBRARY DESTINATION amso)
install(TARGETS amso_core LIBRARY DESTINATION amso)

# Install the pure Python files
install(DIRECTORY amso/ DESTINATION amso)
