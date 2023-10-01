cmake_minimum_required(VERSION 3.16.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the C++ flags
string(APPEND CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -g")

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
include_directories(../../../src/serial/)

# Compile
add_executable(MatrixSum.out ./MatrixSum.cc)
target_include_directories(MatrixSum.out SYSTEM PRIVATE ${doctest_SOURCE_DIR}/doctest)
add_test(NAME MatrixSum.out COMMAND MatrixSum.out)
add_executable(MatrixTransposition.out ./MatrixTransposition.cc)
target_include_directories(MatrixTransposition.out SYSTEM PRIVATE ${doctest_SOURCE_DIR}/doctest)
add_test(NAME MatrixTransposition.out COMMAND MatrixTransposition.out)
add_executable(MatrixMultiplication.out ./MatrixMultiplication.cc)
target_include_directories(MatrixMultiplication.out SYSTEM PRIVATE ${doctest_SOURCE_DIR}/doctest)
add_test(NAME MatrixMultiplication.out COMMAND MatrixMultiplication.out)
