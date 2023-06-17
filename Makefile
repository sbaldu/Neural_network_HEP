export CXX := g++
CXXFLAGS = -O3 -std=c++20

MatrixTestPath = "test/MatrixTest/"
ActivatorsTestPath = "test/ActivatorsTest/"
NetworkTestPath = "test/NetworkTest/"


test: ./test/MatrixTest/* ./test/ActivatorsTest/* ./test/NetworkTest/*
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

	# Test the neural network classes with basic functions
	cd ./test/NetworkTest/ ;
	$(CXX) $(CXXFLAGS) ./test/NetworkTest/OrFunction.cc -o $(NetworkTestPath)Or.out ;
	$(CXX) $(CXXFLAGS) ./test/NetworkTest/AndFunction.cc -o $(NetworkTestPath)And.out ;
	$(CXX) $(CXXFLAGS) ./test/NetworkTest/XorFunction.cc -o $(NetworkTestPath)Xor.out ;

	touch NetworkTest.sh ; chmod +x NetworkTest.sh ;
	echo "#" > NetworkTest.sh ;
	echo "" >> NetworkTest.sh ;
	echo "$(NetworkTestPath)Or.out" >> NetworkTest.sh ;
	echo "$(NetworkTestPath)And.out" >> NetworkTest.sh ;
	echo "$(NetworkTestPath)Xor.out" >> NetworkTest.sh ;

	clear ;
	
clean:
	rm -f *.out
	clear
