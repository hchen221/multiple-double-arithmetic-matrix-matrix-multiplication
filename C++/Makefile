all: matmatTC

matmatTC:
	g++ -c floatybits.cpp TCcuda.cu
	g++ -c floatybits.cpp TCtest.cu
	g++ -c TCcuda.cu TCtest.cu

	g++ floatybits.o TCcuda.o TCtest.o
