
BASE_DIR := ../../../
SRC_DIR := $(BASE_DIR)src/
CUDA_DIR := $(SRC_DIR)cuda/

export CXX := g++
export CUDA := nvcc
CXXFLAGS = -O3 -std=c++20 -I$(CUDA_DIR) -g
CUDAFLAGS = -x cu

all: ./OrFunction.cc ./AndFunction.cc ./XorFunction.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./OrFunction.cc -o Or.out ;
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./AndFunction.cc -o And.out ;
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./XorFunction.cc -o Xor.out ;

	touch Test.sh ; chmod +x Test.sh ;
	echo "#" > Test.sh ;
	echo "" >> Test.sh ;
	echo "./Or.out" >> Test.sh ;
	echo "./And.out" >> Test.sh ;
	echo "./Xor.out" >> Test.sh ;
	clear;

or: ./OrFunction.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./OrFunction.cc -o Or.out ;
	clear;

and: ./AndFunction.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./AndFunction.cc -o And.out ;
	clear;

xor: ./XorFunction.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./XorFunction.cc -o Xor.out ;
	clear;
	
clean:
	rm -f *.out
	clear
