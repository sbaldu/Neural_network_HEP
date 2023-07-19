
BASE_DIR := ../../../../
SRC_DIR := $(BASE_DIR)src/
CUDA_DIR := $(SRC_DIR)cuda/

export CXX := g++
export CUDA := nvcc
CXXFLAGS = -O3 -std=c++20 -I$(CUDA_DIR) -g
CUDAFLAGS = -x cu

all: ./MnistTest.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./MnistTest.cc -o mnist.out ;
	clear

clean:
	rm -f *.out
	clear
