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
    wmma::fragment<wmma::matrix_b, 8,8,4, double, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, 8,8,4, double> C_frag;
    wmma::fill_fragment(C_frag,0.0);

    int I = gridDim.x*blockIdx.x+threadIdx.x;
    int J = gridDim.y*blockIdx.y+threadIdx.y;
    for (int K=0;K<N;K+=4) {
	wmma::load_matrix_sync(A_frag,A+I*N+K,8);
        wmma::load_matrix_sync(B_frag,B+K*N+J,4);

	wmma::mma_sync(C_frag,A_frag,B_frag,C_frag);
    }
    wmma::store_matrix_sync(C+I*N+J,C_frag,8,wmma::mem_row_major);
}

int main() {
    int N = 16;
    vector<double> A;
    vector<double> B;
    vector<double> C;

    for (int i=0;i<N*N;i++) {
        A.push_back(i%N);
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

    dim3 grid(8,4);
    dim3 blox(N/8,N/4);
    mult<<<grid,blox>>>(A_d,B_d,C_d,N);

    cudaMemcpy(C.data(),C_d,N*N*sizeof(double),cudaMemcpyDeviceToHost);
    
    double max_diff = 0;
    for (int i=0;i<N*N;i++) {
	cout << C[i] << " ";
	if (i%N==N-1) {
	    cout << endl;
	}
	if (max_diff<abs(A[i]-C[i])) {
	    max_diff = abs(A[i]-C[i]);
	}
    }
    cout << "Max diff? " << max_diff << endl;

    return 0;
}
