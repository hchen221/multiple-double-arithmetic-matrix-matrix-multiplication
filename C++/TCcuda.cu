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
zeros(m,n,p) returns an mxn matrix of p-double entries with all 0's
*/
vector<double> zeros(int m,int n, int p) {
    vector<double> A;
    for (int i=0;i<m*n*p;i++) {
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
    for (int c=0;c<n;c++) {
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
    vector<double> C = zeros(n,n,p);
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

