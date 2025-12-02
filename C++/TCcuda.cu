#include "TCcuda.h"
#include "floatybits.h"
#include <stdio.h>
#include <iostream>

/*
Assume matrices have a flat representation by default
mat(n,p) returns a flattened nxn matrix of random p doubles
*/
vector<double> mat(int n, int p, int expmin, int expmax) {
    vector<double> A;
    for (int i=0;i<n*n;i++) {
        vector<double> Ai = random_pd(expmin,expmax,p);
        A.insert(A.end(),Ai.begin(),Ai.end());
    }
    return A;
}

/*
zeros(n,p) returns an nxn matrix of p-double entries with all 0's
*/
vector<double> zeros(int n, int p) {
    vector<double> A;
    for (int i=0;i<n*n*p;i++) {
        A.push_back(0);
    }
    return A;
}

// convmult executed on pxp blocks of nlen x nlen threads, where n=nlen*nfrag with nfrag=16
__global__ void convmult(double* A,double* B,double* C_aux,int n,int p) {
    
    wmma::fragment<wmma::matrix_a, 8,8,4, double, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, 8,8,4, double, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, 8,8,4, double> C_frag;
    wmma::fill_fragment(C_frag,0.0);
    

    int i = blockIdx.x;
    int j = blockIdx.y;
    int I = threadIdx.x;
    int J = threadIdx.y;
    // Compute the [I,J] block of A_i*B_j

    for (int K=0;K<n/8;K++) {
	// With Tensor Cores, would load fragments and perform the matrix products here
	/*
	for (int x=0;x<8;x++) {
            for (int y=0;y<4;y++) {
		for (int z=0;z<8;z++) {
		    C_aux[i*n*n*p+j*n*n+(8*I+x)*n+(4*J+y)] += A[i*n*n+(8*I+x)*n+(8*K+z)]*B[j*n*n+(8*K+z)*n+(4*J+y)];
		    __syncthreads();
		}
	    }
	}
	*/
	
        wmma::load_matrix_sync(A_frag,A+i*n*n+(I*8)*n+J*8,8);
        wmma::load_matrix_sync(B_frag,B+j*n*n+(I*8)*n+J*4,8);
	__syncthreads();
	// Perform the matrix product
        wmma::mma_sync(C_frag,A_frag,B_frag,C_frag);
	__syncthreads();
	
    }
    // Copy the result back to C[I,J]
    wmma::store_matrix_sync(C_aux+i*n*n*p+j*n*n+(I*8)*n+J*4,C_frag,8,wmma::mem_row_major);
    __syncthreads();
}

__global__ void convadd(double* C,double* C_aux,int n,int p) { // C is n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts form rows form columns)
    int i = blockIdx.x;
    int I = threadIdx.x;
    int J = threadIdx.y;
    for (int j=0;j<p;j++) {
	if (j<=i) {
	    C[i*n*n+I*n+J] += C_aux[j*n*n*p+(i-j)*n*n+I*n+J];
	}
	__syncthreads();
    }
}

/*This function treats the matrices of p-doubles as a vector of the p matrices in order to perform the convolution
 */
vector<double> manualconvmult(vector<double> A,vector<double> B,int n,int p) {
    vector<double> C_aux = zeros(n,p*p);
    vector<double> C = zeros(n,p);
    
    double* A_d;
    double* B_d;
    double* C_aux_d;
    double* C_d;

    cudaMalloc((void**)&A_d,n*n*p*sizeof(double));
    cudaMalloc((void**)&B_d,n*n*p*sizeof(double));
    cudaMalloc((void**)&C_aux_d,n*n*p*p*sizeof(double));
    cudaMalloc((void**)&C_d,n*n*p*sizeof(double));

    cudaMemcpy(A_d,A.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(C_d,C.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(C_aux_d,C_aux.data(),n*n*p*p*sizeof(double),cudaMemcpyHostToDevice);

    dim3 gridSize(p,p);
    dim3 flatSize(p,1);
    dim3 blockSize(n,n);
    int nr = n/8;
    int nc = n/4;
    dim3 tileBlock(nr,nc);

    convmult<<<gridSize,tileBlock>>>(A_d,B_d,C_aux_d,n,p);
    convadd<<<flatSize,blockSize>>>(C_d,C_aux_d,n,p);
    
    cudaMemcpy(C.data(),C_d,n*n*p*sizeof(double),cudaMemcpyDeviceToHost);
    return C;
}

/*This is a naive implementation which computes the convolutions within the kernel, using nxn blocks and px1 threads per block. Use for comparison purposes*/
__global__ void dotconvbutbetter(double* A,double* B,double* C,int n,int p) {
    int I = blockIdx.x;
    int J = blockIdx.y;
    int i = threadIdx.x;
    for (int k=0;k<n;k++) {
        for (int j=0;j<p;j++) {
            double a,b;
            if (j<=i) {
                a = A[j*n*n+I*n+k];
                b = B[(i-j)*n*n+k*n+J];
            } else {
                a = 0;
                b = 0;
            }
	    __syncthreads();
            C[i*n*n+I*n+J] += a*b;
        }
	__syncthreads();
    }
}

vector<double> directdotconv(vector<double> A,vector<double> B, int n, int p) {
    vector<double> C = zeros(n,p);
    double* Ad;
    double* Bd;
    double* Cd;

    cudaMalloc((void**)&Ad,n*n*p*sizeof(double));
    cudaMalloc((void**)&Bd,n*n*p*sizeof(double));
    cudaMalloc((void**)&Cd,n*n*p*sizeof(double));
    
    cudaMemcpy(Ad,A.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,B.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Cd,C.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    

    dim3 gridSize(n,n);
    dim3 blockSize(p,1);
    dotconvbutbetter<<<gridSize,blockSize>>>(Ad,Bd,Cd,n,p);
    
    cudaMemcpy(C.data(),Cd,n*n*p*sizeof(double),cudaMemcpyDeviceToHost);
    return C;
}

