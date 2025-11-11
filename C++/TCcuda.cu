#include "TCcuda.h"
#include "floatybits.h"
#include <stdio.h>
#include <iostream>
//#include <mma.h>

//using namespace nvcuda;

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
    
    __shared__ double AI[16*16];
    __shared__ double BJ[16*16];
    __shared__ double CIJ[16*16];

    // CIJ will be updated as the kernel executes to store the final result of AI*BJ
    for (int i=0;i<16;i++) {
	for (int j=0;j<16;j++) {
	    CIJ[16*i+j] = C[(16*I+i)*n+(16*J+j)];
	}
    }

    for (int K=0;K<n/16;K++) {
        // load in the appropriate fragments into shared memory
	for (int i=0;i<16;i++) {
	    for (int j=0;j<16;j++) {
		AI[16*i+j] = A[(16*I+i)*n+(16*K+j)];
		BJ[16*i+j] = B[(16*K+i)*n+(16*J+j)];
	    }
	}
        
	// Define and load the Tensor Core fragments
        wmma::fragment<wmma::matrix_a,16,16,16,double,wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b,16,16,16,double,wmma::row_major> B_frag;
        wmma::fragment<wmma::accumulator,16,16,16,double> C_frag;
        wmma::load_matrix_sync(A_frag,AI,16);
        wmma::load_matrix_sync(B_frag,BJ,16);
        wmma::load_matrix_sync(C_frag,CIJ,16);

	// Perform the matrix product
        wmma::mma_sync(C_frag,A_frag,B_frag,C_frag);

        // Copy the result back to CIJ
        wmma::store_matrix_sync(CIJ,C_frag,16,wmma::mem_row_major);
    }
    for (int i=0;i<16;i++) {
	for (int j=0;j<16;j++) {
            C[(16*I+i)*n+(16*J+j)] += CIJ[16*i+j];
	}
    }

    printf("    C[I,J] = %f\n",C[(16*I)*n+16*J]);
}
*/

__global__ void badmul(double* A,double* B,double* C, int n) {
    int I = blockIdx.x;
    int J = blockIdx.y;
    printf("    Computing the (%d,%d) block of C\n",I,J);

    __shared__ double AI[16*16];
    __shared__ double BJ[16*16];
    __shared__ double CIJ[16*16];

    // CIJ will be updated as the kernel executes to store the final result of AI*BJ
    for (int i=0;i<16;i++) {
        for (int j=0;j<16;j++) {
            CIJ[16*i+j] = C[(16*I+i)*n+(16*J+j)];
        }
    }

    for (int K=0;K<n/16;K++) {
        // load in the appropriate fragments into shared memory
        for (int i=0;i<16;i++) {
            for (int j=0;j<16;j++) {
                AI[16*i+j] = A[(16*I+i)*n+(16*K+j)];
                BJ[16*i+j] = B[(16*K+i)*n+(16*J+j)];
            }
        }
        
	// execute the matrix multiplication
	for (int x=0;x<16;x++) {
	    for (int y=0;y<16;y++) {
		for (int z=0;z<16;z++) {
		    CIJ[16*x+y] += AI[16*x+z]*BJ[16*z+y];
		}
	    }
	}

    }
    for (int i=0;i<16;i++) {
        for (int j=0;j<16;j++) {
            C[(16*I+i)*n+(16*J+j)] += CIJ[16*i+j];
        }
    }

    printf("    C[I,J] = %f\n",C[(16*I)*n+16*J]);
}

/*convmult executes p^2 threads in a single block where thread (i,j) computes the matrix product of A_i*B_j, where the result gets accumulated to C_aux_{i,j}. n is the size of the matrix, nfrag=16 is the size of the tile for the matrix products, and p is the number of double parts
 */
__global__ void convmult(double* A,double* B,double* C_aux,int n,int p) { // A,B are n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts for rows form columns)
    int i = blockIdx.x;
    int j = blockIdx.y;
    printf("convmult\n");
    // Want to compute A_i[I,:]*B_j[:,J], C_aux_{i,j}[I,J], capital letters denote matrix entries, lowercase denote parts of the p-double
    if (j <= i) {
	extern __shared__ double Ax[];
        extern __shared__ double By[];
        extern __shared__ double Cxy[];

        for (int I=0;I<n;I++) {
	    for (int J=0;J<n;J++) {
	        Ax[I*n+J] = A[I*n*p+J*p+j]; // take the jth part of A
       	        By[I*n+J] = B[I*n*p+J*p+i-j]; // take the i-j th part of B
	        Cxy[I*n+J] = C_aux[I*n*p*p+J*p*p+i*p+j]; // Locate the i,j position in C_aux
		__syncthreads();
    	    }
	    __syncthreads();
        }

        int nlen = n/16;
        dim3 gridSize(nlen,nlen);
	dim3 haha1(1,1);
        badmul<<<gridSize,haha1>>>(Ax,By,Cxy,n);

        for (int I=0;I<n;I++) {
            for (int J=0;J<n;J++) {
                C_aux[I*n*p*p+J*p*p+i*p+j] = Cxy[I*n+J]; // Locate the i,j position in C_aux
		__syncthreads();
            }
	    __syncthreads();
        }
	printf("Filled back C_aux, C_aux_{i,j}[0,0]=%f\n",C_aux[i*p+j]);
    }

}

__global__ void convadd(double* C,double* C_aux,int n,int p) { // C is n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts form rows form columns)
    printf("convadd\n");
    int i = blockIdx.x;
    int I = threadIdx.x;
    int J = threadIdx.y;
    for (int j=0;j<p;j++) {
	if (j<=i) {
	    C[I*n*p+J*p+i] += C_aux[I*n*p*p+J*p*p+j*p+(i-j)];
	} else {
            C[I*n*p+J*p+i] += 0;
	}
	__syncthreads();
    }
    printf("C[I*n*p+J*p+i]=%f\n",C[I*n*p+J*p+i]);

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
    dim3 haha1(1,1);

    convmult<<<gridSize,haha1,n*n*sizeof(double)>>>(A_d,B_d,C_aux_d,n,p);
    convadd<<<flatSize,blockSize>>>(C_aux_d,C_d,n,p);
    
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

