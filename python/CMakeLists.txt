# Define the Python module
pybind11_add_module(_amso module_amso.cpp)

# Link against your main library
target_link_libraries(_amso PUBLIC amso_core)
target_link_libraries(_amso PRIVATE pybind11::pybind11)
set_target_properties(_amso PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Install the compiled Python module
install(TARGETS _amso LIBRARY DESTINATION amso)
