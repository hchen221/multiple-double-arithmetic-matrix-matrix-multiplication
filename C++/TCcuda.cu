#include "TCcuda.h"
#include "floatybits.h"
#include <stdio.h>
#include <iostream>
//#include <mma.h>

//using namespace nvcuda;

const int nfrag = 16;

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

/*
__global__ void matmul(double* A,double* B,double* C,int n) {
    int I = blockIdx.x;
    int J = blockIdx.y;
    printf("    Computing the (%d,%d) block of C\n",I,J);
    
    __shared__ double AI[nfrag*nfrag];
    __shared__ double BJ[nfrag*nfrag];
    __shared__ double CIJ[nfrag*nfrag];

    // CIJ will be updated as the kernel executes to store the final result of AI*BJ
    for (int i=0;i<nfrag;i++) {
	for (int j=0;j<nfrag;j++) {
	    CIJ[nfrag*i+j] = C[(nfrag*I+i)*n+(nfrag*J+j)];
	}
    }

    for (int K=0;K<n/nfrag;K++) {
        // load in the appropriate fragments into shared memory
	for (int i=0;i<nfrag;i++) {
	    for (int j=0;j<nfrag;j++) {
		AI[nfrag*i+j] = A[(nfrag*I+i)*n+(nfrag*K+j)];
		BJ[nfrag*i+j] = B[(nfrag*K+i)*n+(nfrag*J+j)];
	    }
	}
        
	// Define and load the Tensor Core fragments
        wmma::fragment<wmma::matrix_a,nfrag,nfrag,nfrag,double,wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b,nfrag,nfrag,nfrag,double,wmma::row_major> B_frag;
        wmma::fragment<wmma::accumulator,nfrag,nfrag,nfrag,double> C_frag;
        wmma::load_matrix_sync(A_frag,AI,nfrag);
        wmma::load_matrix_sync(B_frag,BJ,nfrag);
        wmma::load_matrix_sync(C_frag,CIJ,nfrag);

	// Perform the matrix product
        wmma::mma_sync(C_frag,A_frag,B_frag,C_frag);

        // Copy the result back to CIJ
        wmma::store_matrix_sync(CIJ,C_frag,nfrag,wmma::mem_row_major);
    }
    for (int i=0;i<nfrag;i++) {
	for (int j=0;j<nfrag;j++) {
            C[(nfrag*I+i)*n+(nfrag*J+j)] += CIJ[nfrag*i+j];
	}
    }

    printf("    C[I,J] = %f\n",C[(nfrag*I)*n+nfrag*J]);
}
*/

// convmult executed on pxp blocks of nlen x nlen threads, where n=nlen*nfrag with nfrag=16
__global__ void convmult(double* A,double* B,double* C_aux,int n,int p) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int I = threadIdx.x;
    int J = threadIdx.y;
    // Compute the [I,J] block of A_i*B_j
    for (int K=0;K<n/nfrag;K++) {
	// With Tensor Cores, would load fragments and perform the matrix products here
	for (int x=0;x<nfrag;x++) {
            for (int y=0;y<nfrag;y++) {
		for (int z=0;z<nfrag;z++) {
		    C_aux[(nfrag*I+x)*n*p*p+(nfrag*J+y)*p*p+i*p+j] += A[(nfrag*I+x)*n*p+(nfrag*K+z)*p+i]*B[(nfrag*K+z)*n*p+(nfrag*J+y)*p+j];
		    __syncthreads();
		}
	    }
	}
    }
}

__global__ void convadd(double* C,double* C_aux,int n,int p) { // C is n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts form rows form columns)
    int i = blockIdx.x;
    int I = threadIdx.x;
    int J = threadIdx.y;
    for (int j=0;j<p;j++) {
	if (j<=i) {
	    C[I*n*p+J*p+i] += C_aux[I*n*p*p+J*p*p+j*p+(i-j)];
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
    int nlen = n/nfrag;
    dim3 tileBlock(nlen,nlen);

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
                a = A[I*n*p+k*p+j];
                b = B[k*n*p+J*p+(i-j)];
            } else {
                a = 0;
                b = 0;
            }
	    __syncthreads();
            C[I*n*p+J*p+i] += a*b;
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

