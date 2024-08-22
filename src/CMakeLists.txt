# Add your source files
set(_soma_cpu_sources platform_status.cpp
  logger.cpp
)

if(ENABLE_CUDA)
set(_soma_cuda_sources platform_status.cu)
set_source_files_properties(${_soma_cuda_sources} PROPERTIES LANGUAGE CUDA)
set(_soma_sources ${_soma_cpu_sources} ${_soma_cuda_sources})
else(ENABLE_CUDA)
set(_soma_sources ${_soma_cpu_sources} )
endif(ENABLE_CUDA)

add_library(SOMA2 STATIC ${_soma_sources})

set_target_properties(SOMA2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(SOMA2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(SOMA2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Link against CUDA if needed
target_link_libraries(SOMA2 CUDA::cudart)
target_link_libraries(SOMA2 CUDA::cuda_driver)
