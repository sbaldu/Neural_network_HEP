cmake_minimum_required(VERSION 3.16.0)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

# Enable CUDA
enable_language(CUDA)

# Require cuda to compile the test file
find_package(CUDA REQUIRED)

# Set the CUDA standard
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the C++ flags
string(APPEND CUDA_NVCC_FLAGS "-O3 -g -std=c++20")

# Set the folder for the executable
set(EXECUTABLE_OUTPUT_PATH ../)

include(FetchContent)
# Get doctest
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG 7b9885133108ae301ddd16e2651320f54cafeba7
)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_Populate(doctest)
endif()

# Set base directory for serial code
include_directories(../../../../src/cuda/)

# Compile
CUDA_ADD_EXECUTABLE(VectorTest.out VectorTest.cu)
target_include_directories(VectorTest.out SYSTEM PRIVATE ${doctest_SOURCE_DIR}/doctest)
add_test(NAME VectorTest.out COMMAND VectorTest.out)
