
BASE_DIR := ../../../
SRC_DIR := $(BASE_DIR)src/
SERIAL_DIR := $(SRC_DIR)serial/

export CXX := g++
CXXFLAGS = -O3 -std=c++20 -I$(SERIAL_DIR)

all: ./*.cc
	$(CXX) $(CXXFLAGS) ./MatrixSum.cc -o MatrixSum.out ;
	$(CXX) $(CXXFLAGS) ./MatrixTransposition.cc -o MatrixTransposition.out ;
	$(CXX) $(CXXFLAGS) ./MatrixMultiplication.cc -o MatrixMultiplication.out ;
	clear ;

sum: ./*.cc
	$(CXX) $(CXXFLAGS) ./MatrixSum.cc -o MatrixSum.out ;
	clear ;

transpose: ./*.cc
	$(CXX) $(CXXFLAGS) ./MatrixTransposition.cc -o MatrixTransposition.out ;
	clear ;

multiply: ./*.cc
	$(CXX) $(CXXFLAGS) ./MatrixMultiplication.cc -o MatrixMultiplication.out ;
	clear ;

clean:
	rm -f *.out
	clear
