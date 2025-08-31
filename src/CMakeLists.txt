# Library Common

set(LIB_NAME GALAXY_SIM)

message(" Génération des exercices")
add_library(${LIB_NAME} SHARED
    "ExportDLL.hpp"

    "Algorithm/AlgorithmInterface.hpp"
    "Algorithm/CudaSimpleAlgorithm.hpp"
    "Algorithm/CudaSimpleAlgorithm.cu"
    "Algorithm/CpuSimpleAlgorithm.hpp"
    "Algorithm/CpuSimpleAlgorithm.cpp"
    "Algorithm/BarnesHutAlgorithm.hpp"
    "Algorithm/BarnesHutAlgorithm.cpp"

    "Algorithm/CudaBarnesHutAlgorithm.cu"
    "Algorithm/CudaBarnesHutAlgorithm.hpp"

    "Algorithm/Nodes/Bbox.hpp"
    "Algorithm/Nodes/ITreeNode.hpp"
    "Algorithm/Nodes/LastTreeNode.hpp"
    "Algorithm/Nodes/LastTreeNodeUsingCuda.cu"
    "Algorithm/Nodes/LastTreeNodeUsingCuda.hpp"
    "Algorithm/Nodes/CudaLastTreeNode.hpp"
    "Algorithm/Nodes/TreeNode.hpp"
    "Algorithm/Nodes/OctreeNode.hpp"
    "Algorithm/Nodes/SedecTreeNode.hpp"
    "Algorithm/Nodes/FirstTreeNode.hpp"
    "Algorithm/Nodes/CudaTreeNode.cu"
    "Algorithm/Nodes/CudaTreeNode.hpp"

    # "Algorithm/CudaGFieldAlgorithm.hpp"
    # "Algorithm/CudaGFieldAlgorithm.cu"

    ###
    "Physics/dVec3.hpp"
    "Physics/Vec3.hpp"
    "Physics/function.hpp"
    "Physics/constant.hpp"
    "Physics/TypeDef.hpp"
    "Physics/Star.hpp"

    ###
    "GalaxyGenerator/GalaxyGenerator.hpp"
    "GalaxyGenerator/GalaxyGeneratorGPU.cu"

    ###
    "Render/ImageGen.hpp"
    "Render/ImageGen.cpp"

    "Utils/chrono.hpp"
 "Algorithm/CudaBarnesHutAlgorithm.cu")

# Get all OpenCV DLLs from the corresponding binary folder
file(GLOB OpenCV_DLLS
    "${EXTERNAL_DIR}/opencv/build/x64/vc16/bin/*.dll"
)

# message(STATUS "OpenCV DLLs to copy||||||||||||||||||||||||||||||||||||:")
# message(STATUS "  ${OpenCV_DLLS}")
foreach(dll ${OpenCV_DLLS})
    get_filename_component(DLL_NAME "${DLL}" NAME)
    configure_file(${dll} "${BINARY_DIR}/${BUILD_TYPE_DIR}/${DLL_NAME}" COPYONLY)
    # add_custom_command(TARGET ${LIB_NAME} POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dll} "${BINARY_DIR}/${BUILD_TYPE_DIR}/${DLL_NAME}"
    #     COMMENT "Copying ${DLL_NAME} to ${BINARY_DIR}/${BUILD_TYPE_DIR}/${DLL_NAME}"
    # )
endforeach()

set_target_properties(${LIB_NAME}
PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

target_include_directories(
    ${LIB_NAME}
    PRIVATE
    ${SRC_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(GALAXY_SIM PRIVATE ${OpenCV_LIBS})

## Export symbols

message(STATUS "  CMAKE_CXX_COMPILER_ID      : ${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    message(STATUS "  CMAKE_CXX_FLAGS      : ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

target_compile_definitions(${LIB_NAME} PRIVATE GALAXY_SIM_COMPILING_DLL)

if(BUILD_TYPE_DIR STREQUAL "debug")
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
          target_compile_options(${LIB_NAME})
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      MESSAGE("GNU")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      MESSAGE("Intel")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
          target_compile_options(${LIB_NAME} PRIVATE)
    endif()

else()
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
          target_compile_options(${LIB_NAME}
        PRIVATE
        -O2)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      MESSAGE("GNU")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      MESSAGE("Intel")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
          target_compile_options(${LIB_NAME}
        PRIVATE
        -O2)
    endif()
endif()

add_subdirectory("Test")
add_subdirectory("Simulation")