
export CXX := g++
export CUDA := nvcc
CXXFLAGS = -O3 -std=c++20

all: ./*.cu
	$(CUDA) $(CXXFLAGS) ./vec.cu -o vec.out ;
	$(CUDA) $(CXXFLAGS) ./matrix.cu -o matrix.out ;
	clear ;
	
clean:
	rm -f *.out
	clear
