cmake_minimum_required(VERSION 3.18)
project(SOMA2 LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "RelWithDebInfo")
endif()

option(ENABLE_CUDA "Enable compilation with CUDA for GPU acceleration" ON)
if(ENABLE_CUDA)
  add_definitions(-DENABLE_CUDA)
endif()

# Enable CPM
include(cmake/CPM.cmake)

# Pybind11
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  VERSION 2.10.0
)

# Google Test
CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  VERSION 1.13.0
)

# CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)

  enable_language(CUDA)
  find_package(CUDAToolkit 12.0 REQUIRED)
else()
  message(STATUS "No CUDA compiler found")
endif()

# Add source files
add_subdirectory(src)

# Add Python binding
add_subdirectory(python)

# Add tests
enable_testing()
add_subdirectory(tests)
