
BASE_DIR := ../../../../
SRC_DIR := $(BASE_DIR)src/
CUDA_DIR := $(SRC_DIR)cuda/

export CXX := g++
export CUDA := nvcc
CXXFLAGS = -O3 -std=c++20 -I$(CUDA_DIR) -g
CUDAFLAGS = -x cu

all: ./Import.cc ./Export.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./Import.cc -o imp.out ;
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./Export.cc -o exp.out ;

	touch Test.sh ; chmod +x Test.sh ;
	echo "#" > Test.sh ;
	echo "" >> Test.sh ;
	echo "./imp.out" >> Test.sh ;
	echo "./exp.out" >> Test.sh ;
	clear;

exp: ./Export.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./Export.cc -o exp.out ;
	clear;

imp: ./Import.cc
	$(CUDA) $(CUDAFLAGS) $(CXXFLAGS) ./Import.cc -o imp.out ;
	clear;

clean:
	rm -f *.out
	clear
