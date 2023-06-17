export CXX := g++
CXXFLAGS = -O3 -std=c++20 -g

test: ./test/ActivatorsTest/*
	# Test the activators
	$(CXX) $(CXXFLAGS) ./test/ActivatorsTest/Activators.cc -o $(ActivatorsTestPath)Activators.out ;
	clear ;
	
clean:
	rm -f *.out
	clear
