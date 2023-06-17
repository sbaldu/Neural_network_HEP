export CXX := g++
CXXFLAGS = -O3 -std=c++20 -g -pg

all: ./OrFunction.cc ./AndFunction.cc ./XorFunction.cc
	$(CXX) $(CXXFLAGS) ./OrFunction.cc -o $(NetworkTestPath)Or.out ;
	$(CXX) $(CXXFLAGS) ./AndFunction.cc -o $(NetworkTestPath)And.out ;
	$(CXX) $(CXXFLAGS) ./XorFunction.cc -o $(NetworkTestPath)Xor.out ;

	touch Test.sh ; chmod +x Test.sh ;
	echo "#" > Test.sh ;
	echo "" >> Test.sh ;
	echo "./Or.out" >> Test.sh ;
	echo "./And.out" >> Test.sh ;
	echo "./Xor.out" >> Test.sh ;
	clear;

or: ./OrFunction.cc
	$(CXX) $(CXXFLAGS) ./OrFunction.cc -o $(NetworkTestPath)Or.out ;
	clear;

and: ./AndFunction.cc
	$(CXX) $(CXXFLAGS) ./AndFunction.cc -o $(NetworkTestPath)And.out ;
	clear;

xor: ./XorFunction.cc
	$(CXX) $(CXXFLAGS) ./XorFunction.cc -o $(NetworkTestPath)Xor.out ;
	clear;
	
clean:
	rm -f *.out
	clear
