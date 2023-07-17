
BASE_DIR := ../
SRC_DIR := ./
SERIAL_DIR := ./serial/

export CXX := g++
CXXFLAGS = -O3 -std=c++20 -I$(SERIAL_DIR) -g

serial: ./main.cc
	$(CXX) $(CXXFLAGS) ./main.cc -o serial.out ;
	clear

clean:
	rm -f *.out
	clear
