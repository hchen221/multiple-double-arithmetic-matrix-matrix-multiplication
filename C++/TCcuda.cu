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

/*bigA(A,n,p) takes an nxn matrix of p-double entries A and returns [A_1,...,A_p] row stacked, formatted row major*/
vector<double> bigA(vector<double> A,int n,int p) {
    vector<double> AA;
    for (int r=0;r<n;r++) {
	for (int i=0;i<p;i++) {
	    for (int c=0;c<n;c++) {
		AA.push_back(A[r*n*p+c*p+i]);
	    }
	}
    }
    return AA;
}
/*bigB(B,n,p) takes an nxn matrix of p-double entries B and returns the following
  [B_1,B_2,...,B_p]
  [0  ,B_1,...,B_{p-1}]
  [.  ,.  ,.  ..  ]
  [.  ,.  ,.  ,B_1]
  formatted col major
 */
vector<double> bigB(vector<double> B,int n,int p) {
    vector<double> BB;
    for (int c=0;c<n,c++) {
	for (int i=0;i<p;i++) {
	    for (int j=0;j<p;j++) {
		if (j<=i) {
		    for (int r=0;r<n;r++) {
		        BB.push_back(B[r*n*p+c*p+(i-j)]);
		    }
		} else {
		    for (int r=0;r<n;r++) {
			BB.push_back(0);
		    }
		}
	    }
	}
    }
    return BB;
}

__global__ void matmul(double* A,double* B,double* C_aux,int n,int p,int I,int J) {
    
    wmma::fragment<wmma::matrix_a, 8,8,4, double, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, 8,8,4, double, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, 8,8,4, double> C_frag;
    wmma::fill_fragment(C_frag,0.0);
    

    int i = blockDim.x*blockIdx.x+threadIdx.x;
    int j = blockDim.y*blockIdx.y+threadIdx.y;

    for (int k=0;k<n;k+=4) {
	// With Tensor Cores, would load fragments and perform the matrix products here
        /*
	for (int l=0;l<4;l++) {
            C_aux[I*n*n*p+J*n*n+i*n+j] += A[I*n*n+i*n+(k+l)]*B[J*n*n+(k+l)*n+j];
	}
        */
	
	wmma::load_matrix_sync(A_frag,A+I*n*n+i*n+k,8);
        wmma::load_matrix_sync(B_frag,B+J*n*n+k*n+j,4);
	__syncthreads();
        // Perform the matrix product
        wmma::mma_sync(C_frag,A_frag,B_frag,C_frag);
        __syncthreads();
        
    }

    wmma::store_matrix_sync(C_aux+I*n*n*p+J*n*n+i*n+j,C_frag,8,wmma::mem_row_major);
    __syncthreads();

}

__global__ void convmult(double* A,double* B,double* C_aux,int n,int p) {
    int I = threadIdx.x;
    int J = threadIdx.y;
    
    dim3 grid(n/8,n/8);
    dim3 blox(8,8);
    
    matmul<<<grid,blox>>>(A,B,C_aux,n,p,I,J);
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
    dim3 hahaONE(1,1);

    convmult<<<hahaONE,gridSize>>>(A_d,B_d,C_aux_d,n,p);
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

