
# Find all test files ending with .cpp that include 'test' in their name
file(GLOB test_sources "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(test_source ${test_sources})
    # Extract the filename without extension to use as the target name
    get_filename_component(test_name ${test_source} NAME_WE)

    # Create an executable for each test source file
    add_executable(${test_name} ${test_source})

    # Link the test executable with necessary libraries
    target_link_libraries(${test_name} PRIVATE SOMA2 gtest gtest_main)

    # Register the test with CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
