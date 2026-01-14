CUDA=/usr/local/cuda/include
GPP=/usr/bin/gcc

all: matmatTC

basic:
	nvcc -ccbin=$(GPP) -arch=sm_80 -c basic.cu
	nvcc -ccbin=$(GPP) basic.o -o basicrun
	./basicrun

blog:
	nvcc -o TCGEMM -arch=sm_80 -lcublas -lcurand simpleTensorCoreGEMM.cu
	./TCGEMM

smTCGemm:
	nvcc -ccbin=/usr/bin/gcc -O3 -o dmmarun -arch=sm_80 \
	dmmatest.cu
	./dmmarun

matmatTC:
	nvcc -ccbin=$(GPP) -rdc=true -arch=sm_80 -c floatybits.cu
	nvcc -ccbin=$(GPP) -rdc=true -arch=sm_80 -c TCcuda.cu
	nvcc -ccbin=$(GPP) -rdc=true -arch=sm_80 -c TCtest.cu
	nvcc -ccbin=$(GPP) -rdc=true floatybits.o TCcuda.o TCtest.o -o mytest
	./mytest

matmatTC2:
	nvcc -c -rdc=true floatybits.cu
	nvcc -c -rdc=true TCcuda.cu
	nvcc -c -rdc=true TCtest.cu
	nvcc -rdc=true floatybits.o TCcuda.o TCtest.o -o mytest
	./mytest

cudantTC:
	gcc -c TCcudant.cpp
	c++ TCcudant.o -o cudant
	./cudant

clean:
	rm -f *.o
	rm mytest
	rm basicrun
	rm cudant
	rm dmmarun
