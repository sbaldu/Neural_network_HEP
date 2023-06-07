export CXX := g++
CXXFLAGS = -std=c++20

MatrixTestPath = "test/MatrixTest/"
ActivatorsTestPath = "test/ActivatorsTest/"

test: ./test/MatrixTest/* ./test/ActivatorsTest/*
	# Test the matrix class
	cd ./test/MatrixTest/ ;
	$(CXX) $(CXXFLAGS) ./test/MatrixTest/MatrixSum.cc -o $(MatrixTestPath)MatrixSum.out ;
	$(CXX) $(CXXFLAGS) ./test/MatrixTest/MatrixTransposition.cc -o $(MatrixTestPath)MatrixTransposition.out ;
	$(CXX) $(CXXFLAGS) ./test/MatrixTest/MatrixMultiplication.cc -o $(MatrixTestPath)MatrixMultiplication.out ;
	touch MatrixTest.sh ; chmod +x MatrixTest.sh ;
	echo "#" > MatrixTest.sh ;
	echo "" >> MatrixTest.sh ;
	echo "./$(MatrixTestPath)MatrixSum.out" >> MatrixTest.sh ;
	echo "./$(MatrixTestPath)MatrixTransposition.out" >> MatrixTest.sh ;
	echo "./$(MatrixTestPath)MatrixMultiplication.out" >> MatrixTest.sh ;

	# Test the activators
	cd ./test/ActivatorsTest/ ;
	$(CXX) $(CXXFLAGS) ./test/ActivatorsTest/Activators.cc -o $(ActivatorsTestPath)Activators.out ;

	touch ActivatorsTest.sh ; chmod +x ActivatorsTest.sh ;
	echo "#" > ActivatorsTest.sh ;
	echo "" >> ActivatorsTest.sh ;
	echo "$(ActivatorsTestPath)Activators.out" >> ActivatorsTest.sh ;

	clear ;
	
clean:
	rm -f *.out
	clear
