CUDA=/usr/local/cuda/include

all: matmatTC

basic:
	nvcc -c basic.cu
	nvcc basic.o -o basicrun

matmatTC:
	gcc -c floatybits.cpp
	nvcc -c TCcuda.cu
	nvcc -c TCtest.cu
	nvcc floatybits.o TCcuda.o TCtest.o -o mytest

matmatTC2:
	gcc -c floatybits.cpp
	nvcc -c TCcuda.cu
	nvcc -c TCtest2.cu
	nvcc floatybits.o TCcuda.o TCtest2.o -o mytest

cudantTC:
	gcc -c TCcudant.cpp
	c++ TCcudant.o -o cudant

clean:
	rm -f *.o
	rm mytest
	rm cudant
	rm basicrun
