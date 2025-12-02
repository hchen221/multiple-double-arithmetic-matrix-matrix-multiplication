CUDA=/usr/local/cuda/include
GPP=/usr/bin/gcc

all: matmatTC

basic:
	nvcc -c basic.cu
	nvcc basic.o -o basicrun
	./basicrun

matmatTC:
	nvcc -ccbin=$(GPP) -arch=sm_80 -c floatybits.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCcuda.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCtest.cu
	nvcc -ccbin=$(GPP) floatybits.o TCcuda.o TCtest.o -o mytest
	./mytest

matmatTC2:
	gcc -c floatybits.cpp
	nvcc -c TCcuda.cu
	nvcc -c TCtest2.cu
	nvcc floatybits.o TCcuda.o TCtest2.o -o mytest
	./mytest

cudantTC:
	gcc -c TCcudant.cpp
	c++ TCcudant.o -o cudant
	./cudant

clean:
	rm -f *.o
	rm mytest
	rm cudant
	rm basicrun
