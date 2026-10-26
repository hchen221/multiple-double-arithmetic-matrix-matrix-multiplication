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
	nvcc -ccbin=$(GPP) -arch=sm_80 -c floatybits.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c double_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c quad_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c octo_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c hexa_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCcuda.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCtest.cu
	nvcc -ccbin=$(GPP) floatybits.o TCcuda.o TCtest.o -o mytest
	./mytest

mini:
	nvcc -ccbin=$(GPP) -arch=sm_80 -c floatybits.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c double_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c quad_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c octo_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c hexa_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCcuda.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCtest_mini.cu
	nvcc -ccbin=$(GPP) floatybits.o TCcuda.o TCtest_mini.o -o minitest
	./minitest

invdiag:
	nvcc -ccbin=$(GPP) -arch=sm_80 -c floatybits.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c double_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c quad_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c octo_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c hexa_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCcuda.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCtest_inv_diag.cu
	nvcc -ccbin=$(GPP) floatybits.o TCcuda.o TCtest_mini.o -o invdiagtest
	./invdiagtest

dvcconv:
	nvcc -ccbin=$(GPP) -arch=sm_80 -c floatybits.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c double_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c quad_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c octo_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c hexa_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCcuda.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCtest_dvc_conv.cu
	nvcc -ccbin=$(GPP) floatybits.o TCcuda.o TCtest_dvc_conv.o -o dvcconvtest
	./dvcconvtest

nh:
	nvcc -ccbin=$(GPP) -arch=sm_80 -c floatybits.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c double_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c quad_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c octo_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c hexa_double_functions.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCcuda.cu
	nvcc -ccbin=$(GPP) -arch=sm_80 -c TCtest.cu
	nvcc -ccbin=$(GPP) floatybits.o TCcuda.o TCtest.o -o mytest
	nohup ./mytest &

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
