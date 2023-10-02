cmake_minimum_required(VERSION 3.16.0)

project(Cuda_tests VERSION 1.0.0 LANGUAGES CXX CUDA)

add_subdirectory(./KernelTest/)
add_subdirectory(./MatrixTest/)
add_subdirectory(./VectorTest/)
