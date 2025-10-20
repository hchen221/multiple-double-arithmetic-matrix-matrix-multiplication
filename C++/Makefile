all: matmatTC

matmatTC:
    gcc -c floatybits.cpp
	nvcc -c TCcuda.cu
    gcc TCtest.cpp
