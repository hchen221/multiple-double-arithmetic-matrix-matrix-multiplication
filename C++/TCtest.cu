#include "TCcuda.h"
#include <iostream>
#include <cmath>
using namespace std;

#define p 2
#define n 256

#define M 8
#define N 8
#define K 4

#define M_GLOBAL n
#define N_GLOBAL n*4*p
#define K_GLOBAL n*4*p

__global__ void matmul(double *a, double *b, double *c) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

    int cCol = warpN * N;
    int cRow = warpM * M;

    wmma::load_matrix_sync(c_frag,c+cCol+cRow*N_GLOBAL,N_GLOBAL,wmma::mem_row_major);

    for (int k=0;k<K_GLOBAL;k+=K) {
	int aCol = k;
        int aRow = warpM * M;
        int bCol = warpN * N;
        int bRow = k;
	wmma::load_matrix_sync(a_frag, a + aCol + aRow * K_GLOBAL, K_GLOBAL);
        wmma::load_matrix_sync(b_frag, b + bRow + bCol * K_GLOBAL, K_GLOBAL);

	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(c+cCol+cRow*N_GLOBAL,c_frag,N_GLOBAL,wmma::mem_row_major);
}

void test(int expmin,int expmax) {
    vector<double> A = mat(n,p,expmin,expmax);
    vector<double> B = mat(n,p,expmin,expmax);
    vector<double> A8 = split4pd(A);
    vector<double> B8 = split4pd(B);
    vector<double> A8D = bigA(A8,n,4*p);
    vector<double> B8D = bigB(B8,n,4*p);
    cout << "A,B in R^{" << n << "x" << n << "}, entries of "<< p << "-doubles\n\n";
    
    vector<double> C1 = zeros(n,n,4*p);

    cout << "Setup? Done." << endl;

    double* A_d;
    double* B_d;
    double* C_d;

    cudaMalloc((void**)&A_d,M_GLOBAL*K_GLOBAL*sizeof(double));
    cudaMemcpy(A_d,A8D.data(),M_GLOBAL*K_GLOBAL*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d,K_GLOBAL*N_GLOBAL*sizeof(double));
    cudaMemcpy(B_d,B8D.data(),K_GLOBAL*N_GLOBAL*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d,M_GLOBAL*N_GLOBAL*sizeof(double));
    cudaMemcpy(C_d,C1.data(),M_GLOBAL*N_GLOBAL*sizeof(double),cudaMemcpyHostToDevice);

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize 128x4 means we have
    // 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (M*blockDim.x/32-1))/(M*blockDim.x/32);
    gridDim.y = (N_GLOBAL + N*blockDim.y-1)/(N*blockDim.y);

    matmul<<<gridDim,blockDim>>>(A_d,B_d,C_d);

    cudaMemcpy(C1.data(),C_d,M_GLOBAL*N_GLOBAL*sizeof(double),cudaMemcpyDeviceToHost);

    cout << "Device? Finished." << endl;

    vector<double> C2 = bigA(directdotconv(A8,B8,n,4*p),n,4*p);
    cout << "Host? Finished." << endl;
    
    cout << "Device C[1,1]? (";
    for (int i=0;i<4*p;i++) {
        cout << C1[i*n];
        if (i<4*p-1) {
            cout << ",";
        }
    }
    cout << ")" << endl;

    cout << "Host C[1,1]? (";
    for (int i=0;i<4*p;i++) {
        cout << C2[i*n];
	if (i<4*p-1) {
	    cout << ",";
	}
    }
    cout << ")" << endl;
    
    double max_err = 0;
    for (int i=0;i<n*n*4*p;i++) {
	if (max_err<abs(C1[i]-C2[i])) {
	    max_err = abs(C1[i]-C2[i]);
	}
    }
    cout << "Max error? " << max_err << endl;
}

int main() {
    test(0,0);
    return 0;
}



