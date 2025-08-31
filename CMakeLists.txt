cmake_minimum_required(VERSION 3.17)

# Configuration

set(USE_CUDA OFF
       CACHE BOOL "Compiler code CUDA.")
set(BUILD_UNIT_TESTS ON
       CACHE BOOL "Build des tests unitaires.")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

### Test data directory relative to unit test exe file

# Extra Debug or Release folder when compiled with MSVC
file(RELATIVE_PATH TEST_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/Release ${PROJECT_SOURCE_DIR}/data)


### External libraries folders management
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BUILD_TYPE_DIR "debug")
else()
    set(BUILD_TYPE_DIR "release")
endif()

set(OS_DIR "windows")


### CUDA
set (GS_LANGUAGES "C;CXX")
if(USE_CUDA)
    list(APPEND GS_LANGUAGES "CUDA")

    set(CUDATOOLKITDIR  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
    set(CMAKE_VS_PLATFORM_TOOLSET_CUDA 12.9)
    set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe")

    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES all)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_COMPILE_OPTIONS "-O2 -m64 -dc")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GALAXY_SIM
        LANGUAGES ${GS_LANGUAGES}
        VERSION "3.0")


#### Project Directories

set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")
set(TEST_DATA_DIR "${PROJECT_SOURCE_DIR}/test_data")
set(BINARY_DIR "${CMAKE_BINARY_DIR}/bin")
set(OUTPUT_DATA "${PROJECT_SOURCE_DIR}/test_data/output")

# Find OpenCV
set(OpenCV_DIR "${EXTERNAL_DIR}/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# création du dossier de résultats si non existant
make_directory(${OUTPUT_DATA})

include(FetchContent)
#### Google test
if(BUILD_UNIT_TESTS)
    # Version 1.12.1 is the last compatible with C++11
    FetchContent_Declare(
    googletest
    URL "${EXTERNAL_DIR}/gtest/googletest-release-1.12.1.zip"
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

#### Source subdirectory
add_subdirectory(src)