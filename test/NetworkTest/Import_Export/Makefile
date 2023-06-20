export CXX := g++
CXXFLAGS = -O3 -std=c++20 -g

all: ./Import.cc ./Export.cc
	$(CXX) $(CXXFLAGS) ./Import.cc -o imp.out ;
	$(CXX) $(CXXFLAGS) ./Export.cc -o exp.out ;

	touch Test.sh ; chmod +x Test.sh ;
	echo "#" > Test.sh ;
	echo "" >> Test.sh ;
	echo "./imp.out" >> Test.sh ;
	echo "./exp.out" >> Test.sh ;
	clear;

exp: ./Export.cc
	$(CXX) $(CXXFLAGS) ./Export.cc -o exp.out ;
	clear;

imp: ./Import.cc
	$(CXX) $(CXXFLAGS) ./Import.cc -o imp.out ;
	clear;

clean:
	rm -f *.out
	clear
