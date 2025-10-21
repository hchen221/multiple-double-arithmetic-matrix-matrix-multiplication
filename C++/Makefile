CUDA=/usr/local/cuda/include

all: matmatTC

matmatTC:
	gcc -c floatybits.cpp
	nvcc -c TCcuda.cu
	nvcc -c TCtest.cu
	nvcc floatybits.o TCcuda.o TCtest.o -o mytest

cudantTC:
	gcc -c TCcudant.cpp
	c++ TCcudant.o -o cudant

clean:
	rm -f *.o
	rm mytest
	rm cudant
