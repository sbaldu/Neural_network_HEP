
BASE_DIR := ../../../../
SRC_DIR := $(BASE_DIR)src/
SERIAL_DIR := $(SRC_DIR)serial/

export CXX := g++
CXXFLAGS = -O3 -std=c++20 -I$(SERIAL_DIR) -g

all: ./MnistTest.cc
	$(CXX) $(CXXFLAGS) ./MnistTest.cc -o a.out ;
	clear

clean:
	rm -f *.out
	clear
