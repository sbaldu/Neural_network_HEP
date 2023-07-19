
BASE_DIR := ../../../../
SRC_DIR := $(BASE_DIR)src/
CUDA_DIR := $(SRC_DIR)cuda/

export CXX := g++
export CUDA := nvcc
CXXFLAGS = -O3 -std=c++20 -I$(CUDA_DIR)

all: ./*.cu
	$(CUDA) $(CXXFLAGS) ./vec.cu -o vec.out ;
	$(CUDA) $(CXXFLAGS) ./matrix.cu -o matrix.out ;
	clear ;

vec: ./vec.cu
	$(CUDA) $(CXXFLAGS) ./vec.cu -o vec.out ;
	clear ;

matrix: ./matrix.cu
	$(CUDA) $(CXXFLAGS) ./matrix.cu -o matrix.out ;
	clear ;
	
clean:
	rm -f *.out
	clear
