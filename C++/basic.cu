#include <iostream>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
using namespace std;

__global__ void mult(double* A,double* B,double* C,int N) {
    wmma::fragment<wmma::matrix_a, 8,8,4, double, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, 8,8,4, double, wmma::col_major> B_frag;
    wmma::fragment<wmma::accumulator, 8,8,4, double> C_frag;
    
    //printf("%i\n",warpSize);

    int I = (blockDim.x*blockIdx.x+threadIdx.x)/warpSize;
    int J = blockDim.y*blockIdx.y+threadIdx.y;
    
    wmma::load_matrix_sync(C_frag,C+I*N+J,N,wmma::mem_row_major);
    
    for (int K=0;K<N;K+=4) {
	wmma::load_matrix_sync(A_frag,A+I*8*N+K,N);
        wmma::load_matrix_sync(B_frag,B+J*8*N+K,N);
	__syncthreads();
	wmma::mma_sync(C_frag,A_frag,B_frag,C_frag);
	__syncthreads();
    }
    wmma::store_matrix_sync(C+I*8*N+J*8,C_frag,N,wmma::mem_row_major);
}

int main() {
    int N = 2048;
    vector<double> A;
    vector<double> B;
    vector<double> C;

    for (int i=0;i<N*N;i++) {
        A.push_back(i%N+1);
	C.push_back(0);
	if (i%N==(i%(N*N))/N) {
	    B.push_back(1);
	} else {
	    B.push_back(0);
	}
    }

    double* A_d;
    double* B_d;
    double* C_d;

    cudaMalloc((void**)&A_d,N*N*sizeof(double));
    cudaMemcpy(A_d,A.data(),N*N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d,N*N*sizeof(double));
    cudaMemcpy(B_d,B.data(),N*N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d,N*N*sizeof(double));
    cudaMemcpy(C_d,C.data(),N*N*sizeof(double),cudaMemcpyHostToDevice);
    
    dim3 grid;
    dim3 blox(128,4);
    grid.x = (N+(8*128/32-1))/(8*128/32);
    grid.y = (N+(8*4-1))/(8*4);
    
    cout << grid.x << ","<< grid.y << endl;

    mult<<<grid,blox>>>(A_d,B_d,C_d,N);
    
    cudaMemcpy(C.data(),A_d,N*N*sizeof(double),cudaMemcpyDeviceToHost);
    
    double max_diff = 0;
    
    for (int i=0;i<N*N;i++) {
	//cout << C[i] << " ";
	if (i%N==N-1) {
	    //cout << endl;
	}
	if (max_diff<abs(A[i]-C[i])) {
	    max_diff = abs(A[i]-C[i]);
	}
    }
    
    cout << "Max diff? " << max_diff << endl;

    return 0;
}
