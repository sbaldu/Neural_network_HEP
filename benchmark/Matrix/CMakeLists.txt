cmake_minimum_required(VERSION 3.16.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the C++ flags
string(APPEND CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -g")

# Set the folder for the executable
set(EXECUTABLE_OUTPUT_PATH ../../)

# Set base directory for serial data formats
include_directories(../../src/serial/DataFormats)
include_directories(..)

# Compile
add_executable(MatrixOperations.out ./MatrixOperations.cc)
add_executable(MatrixConstruction.out ./MatrixConstruction.cc)
