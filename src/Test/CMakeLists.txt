                                                                                                            set(TEST_EXE_NAME GALAXY_SIM_TEST)

add_executable(
    ${TEST_EXE_NAME}
    "test.cpp"
    "GenImageTest.cpp" 
    "Simulate.cpp" 
    "BenchMark.cpp" 
    "BarnesHutTest.cpp")

set_target_properties(${TEST_EXE_NAME}
PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

target_include_directories(
    ${TEST_EXE_NAME}
    PRIVATE ${SRC_DIR}
    )

target_link_libraries(
    ${TEST_EXE_NAME}
    PRIVATE
    gtest_main
    GALAXY_SIM
    ${OpenCV_LIBS}
)

## Export symbols

target_compile_definitions(${TEST_EXE_NAME} PRIVATE GALAXY_SIM_COMPILING_DLL)

include(GoogleTest)
gtest_discover_tests(${TEST_EXE_NAME} DISCOVERY_TIMEOUT 10)