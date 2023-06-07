export CXX := g++
CXXFLAGS = -std=c++20

MatrixTestPath = "test/MatrixTest/"

test: ./test/MatrixTest/*
	cd ./test/MatrixTest/ ;
	$(CXX) $(CXXFLAGS) ./test/MatrixTest/MatrixSum.cc -o $(MatrixTestPath)MatrixSum.out ;
	$(CXX) $(CXXFLAGS) ./test/MatrixTest/MatrixTransposition.cc -o $(MatrixTestPath)MatrixTransposition.out ;
	$(CXX) $(CXXFLAGS) ./test/MatrixTest/MatrixMultiplication.cc -o $(MatrixTestPath)MatrixMultiplication.out ;
	touch MatrixTest.sh ; chmod +x MatrixTest.sh ;
	echo "#" >> MatrixTest.sh ;
	echo "" >> MatrixTest.sh ;
	echo "$(MatrixTestPath)MatrixSum.out" >> MatrixTest.sh ;
	echo "$(MatrixTestPath)MatrixTransposition.out" >> MatrixTest.sh ;
	echo "$(MatrixTestPath)MatrixMultiplication.out" >> MatrixTest.sh ;

	clear ;
	
clean:
	rm -f *.out
	clear
