                                                                                                            set(EXE_MAIN GALAXY_SIM_TEST)
set(EXE_MAIN Simulation)
add_executable(
    ${EXE_MAIN}
    "Simulate.cpp"
    "TimeCompute.hpp"
)

set_target_properties(${EXE_MAIN}
PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

target_include_directories(
    ${EXE_MAIN}
    PRIVATE ${SRC_DIR}
    )

target_link_libraries(
    ${EXE_MAIN}
    PRIVATE
    GALAXY_SIM
)

## Export symbols

target_compile_definitions(${EXE_MAIN} PRIVATE GALAXY_SIM_COMPILING_DLL)