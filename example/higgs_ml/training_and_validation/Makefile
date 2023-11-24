
BASE_DIR := ../
SRC_DIR := ./
SERIAL_DIR := ./serial/

export CXX := g++
CXXFLAGS = -O3 -Wall -std=c++20 -I$(SERIAL_DIR) -g

serial: ./main.cc
	$(CXX) $(CXXFLAGS) ./training_and_validation.cc -o tv.out ;
	$(CXX) $(CXXFLAGS) ./test.cc -o test.out ;

clean:
	rm -f *.out
	clear
