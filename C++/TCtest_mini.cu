#include "TCcuda.h"
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

#define M 8
#define N 8
#define K 4

#define n_mini 16
#define b_mini 4
/*
   matmul_mini is a basic Tensor Core matrix product AB=C where A,B,C are n_mini x n_mini matrices
   b_mini is a block size. We change the range of the parameters in the kernel (i.e. k) to see if we can multiply only a subset of the matrix (i.e. the first b_mini x b_mini block of C only)
*/
__global__ void matmul_mini(double *a, double *b, double *c) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

    int cCol = warpN * N;
    int cRow = warpM * M;

    wmma::load_matrix_sync(c_frag,c+cCol+cRow*n_mini,n_mini,wmma::mem_row_major);
    //The range on k determines which columns are multiplied. How do you restrict rows?
    for (int k=0;k<b_mini;k+=K) {
        int aCol = k;
        int aRow = warpM * M;
        int bCol = warpN * N;
        int bRow = k;
        wmma::load_matrix_sync(a_frag, a + aCol + aRow * n_mini, n_mini);
        wmma::load_matrix_sync(b_frag, b + bRow + bCol * n_mini, n_mini);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(c+cCol+cRow*n_mini,c_frag,n_mini,wmma::mem_row_major);
}
/*test_mini is a driver program to test the above kernel with A as all 1's and B as 2 times the identity matrix*/
void test_mini() {
    vector<double> A = ones(n_mini,n_mini,1);
    vector<double> B = zeros(n_mini,n_mini,1);
    for (int i=0;i<n_mini;i++) {
        B[i*n_mini+i] = 2;
    }
    cout << "A=" << endl;
    for (int i=0;i<n_mini*n_mini;i++) {
        if (i%n_mini==0) {
            cout << endl;
        }
        cout << A[i] << " ";
    }
    cout << endl << endl;
    cout << "B=" << endl;
    for (int i=0;i<n_mini*n_mini;i++) {
        if (i%n_mini==0) {
            cout << endl;
        }
        cout << B[i] << " ";
    }
    cout << endl << endl;
    vector<double> C = zeros(n_mini,n_mini,1);
    double* A_d;
    double* B_d;
    double* C_d;
    cudaMalloc((void**)&A_d,(long long int)n_mini*(long long int)n_mini*(long long int)sizeof(double));
    cudaMemcpy(A_d,A.data(),(long long int)n_mini*(long long int)n_mini*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d,(long long int)n_mini*(long long int)n_mini*(long long int)sizeof(double));
    cudaMemcpy(B_d,B.data(),(long long int)n_mini*(long long int)n_mini*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d,(long long int)n_mini*(long long int)n_mini*(long long int)sizeof(double));
    cudaMemcpy(C_d,C.data(),(long long int)n_mini*(long long int)n_mini*(long long int)sizeof(double),cudaMemcpyHostToDevice);

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize 128x4 means we have
    // 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    // Does changing n_mini to b_mini here control the rows?
    gridDim.x = (n_mini + (M*blockDim.x/32-1))/(M*blockDim.x/32);
    gridDim.y = (n_mini + N*blockDim.y-1)/(N*blockDim.y);

    matmul_mini<<<gridDim,blockDim>>>(A_d,B_d,C_d);
    cudaMemcpy(C.data(),C_d,(long long int)n_mini*(long long int)n_mini*(long long int)sizeof(double),cudaMemcpyDeviceToHost);
    cout << "C=" << endl;
    for (int i=0;i<n_mini*n_mini;i++) {
        if (i%n_mini==0) {
            cout << endl;
        }
        cout << C[i] << " ";
    }
}

int main() {
    test_mini();
    return 0;
}
