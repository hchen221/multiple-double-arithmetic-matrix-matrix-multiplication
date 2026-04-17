#include "TCcuda.h"
#include "floatybits.h"
#include <stdio.h>
#include <iostream>
#include "double_double_functions.cu"
#include "quad_double_functions.cu"
#include "octo_double_functions.cu"
#include "hexa_double_functions.cu"

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

vector<double> part(vector<double> x,int p,int i) {
    vector<double> xi;
    for (int j=i;j<x.size();j+=p) {
        xi.push_back(x[j]);
    }
    return xi;
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
    for (int i=0;i<p;i++) {
    for (int c=0;c<n;c++) {
    for (int j=0;j<p;j++) {
    for (int r=0;r<n;r++) {
        if (j<=i) {
            BB.push_back(B[r*n*p+c*p+(i-j)]);
        } else {
            BB.push_back(0);
        }
    }
    }
    }
    }
    return BB;
}

vector<double> bigB2(vector<double> B,int n, int p) {
    vector<double> BB;
    for (int c=0;c<n;c++) {
    for (int i=0;i<p;i++) {
    for (int r=0;r<n;r++) {
    for (int j=0;j<p;j++) {
        if (j<=i) {
	    BB.push_back(B[r*n*p+c*p+(i-j)]);
	} else {
	    BB.push_back(0);
        }
    }
    }
    }
    }
    return BB;
}

__global__ void renormbigA(double* A,int n,int p) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    for (int k=0;k<p-1;k++) {
	double newhi = A[i*n*p+k*n+j]+A[i*n*p+(k+1)*n+j];
	double newlo = newhi-A[i*n*p+k*n+j];
	A[i*n*p+k*n+j] = newhi;
	A[i*n*p+(k+1)*n+j] = newlo;
    }
}

__global__ void renormA(double* A,int n,int p) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    for (int k=0;k<p-1;k++) {
        double newhi = A[i*n*p+j*p+k]+A[i*n*p+j*p+(k+1)];
        double newlo = newhi-A[i*n*p+j*p+k];
        A[i*n*p+j*p+k] = newhi;
        A[i*n*p+j*p+(k+1)] = newlo;
    }
}

__global__ void ddmm(double *A,double *B,double *C,int n)
{
    int r = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;

    double prd1 = 0.0;
    double prd2 = 0.0;

    double a1,a2,b1,b2,c1,c2;
    for(int k=0; k<n; k++)
    {
        a1 = A[r*n*2+k*2]; a2 = A[r*n*2+k*2+1];
        b1 = B[k*n*2+c*2]; b2 = B[k*n*2+c*2+1];
        
        ddf_mul(a1,a2,b1,b2,&c1,&c2);
        ddf_add(prd1,prd2,c1,c2,&prd1,&prd2);
    }
    C[r*n*2+c*2] = prd1;
    C[r*n*2+c*2+1] = prd2;
}

__global__ void qdmm(double *A,double *B,double *C,int n)
{
    int r = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;

    double prd1 = 0.0;
    double prd2 = 0.0;
    double prd3 = 0.0;
    double prd4 = 0.0;

    double a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;

    for(int k=0; k<n; k++)
    {
        a1 = A[r*n*4+k*4]; a2 = A[r*n*4+k*4+1]; a3 = A[r*n*4+k*4+2]; a4 = A[r*n*4+k*4+3];
        b1 = B[r*n*4+k*4]; b2 = B[r*n*4+k*4+1]; b3 = B[r*n*4+k*4+2]; b4 = B[r*n*4+k*4+3];

        qdf_mul(a1,a2,a3,a4,b1,b2,b3,b4,&c1,&c2,&c3,&c4);
        qdf_add(prd1,prd2,prd3,prd4,c1,c2,c3,c4,&prd1,&prd2,&prd3,&prd4);
    }
    C[r*n*4+c*4] = prd1;
    C[r*n*4+c*4+1] = prd2;
    C[r*n*4+c*4+2] = prd3;
    C[r*n*4+c*4+3] = prd4;
}

__global__ void odmm(double *A,double *B,double *C,int n)
{
    int r = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;

    double prd1 = 0.0;
    double prd2 = 0.0;
    double prd3 = 0.0;
    double prd4 = 0.0;
    double prd5 = 0.0;
    double prd6 = 0.0;
    double prd7 = 0.0;
    double prd8 = 0.0;

    double a1,a2,a3,a4,a5,a6,a7,a8,b1,b2,b3,b4,b5,b6,b7,b8,c1,c2,c3,c4,c5,c6,c7,c8;
    for(int k=0; k<n; k++)
    {
        a1 = A[r*n*8+k*8]; a2 = A[r*n*8+k*8+1]; a3 = A[r*n*8+k*8+2]; a4 = A[r*n*8+k*8+3]; a5 = A[r*n*8+k*8+4]; a6 = A[r*n*8+k*8+5]; a7 = A[r*n*8+k*8+6]; a8 = A[r*n*8+k*8+7];
        b1 = B[k*n*8+c*8]; b2 = B[k*n*8+c*8+1]; b3 = B[k*n*8+c*8+2]; b4 = B[k*n*8+c*8+3]; b5 = B[k*n*8+c*8+4]; b6 = B[k*n*8+c*8+5]; b7 = B[k*n*8+c*8+6]; b8 = B[k*n*8+c*8+7];
        
        odf_mul(a1,a2,a3,a4,a5,a6,a7,a8,b1,b2,b3,b4,b5,b6,b7,b8,&c1,&c2,&c3,&c4,&c5,&c6,&c7,&c8);
        odf_add(prd1,prd2,prd3,prd4,prd5,prd6,prd7,prd8,c1,c2,c3,c4,c5,c6,c7,c8,&prd1,&prd2,&prd3,&prd4,&prd5,&prd6,&prd7,&prd8);
    }
    C[r*n*8+c*8] = prd1;
    C[r*n*8+c*8+1] = prd2;
    C[r*n*8+c*8+2] = prd3;
    C[r*n*8+c*8+3] = prd4;
    C[r*n*8+c*8+4] = prd5;
    C[r*n*8+c*8+5] = prd6;
    C[r*n*8+c*8+6] = prd7;
    C[r*n*8+c*8+7] = prd8;
}

__global__ void hdmm(double *A,double *B,double *C,int n)
{
    int r = blockIdx.x*blockDim.x+threadIdx.x;
    int c = blockIdx.y*blockDim.y+threadIdx.y;

    double prd1 = 0.0;
    double prd2 = 0.0;
    double prd3 = 0.0;
    double prd4 = 0.0;
    double prd5 = 0.0;
    double prd6 = 0.0;
    double prd7 = 0.0;
    double prd8 = 0.0;
    double prd9 = 0.0;
    double prd10 = 0.0;
    double prd11 = 0.0;
    double prd12 = 0.0;
    double prd13 = 0.0;
    double prd14 = 0.0;
    double prd15 = 0.0;
    double prd16 = 0.0;

    double a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16;
    for(int k=0; k<n; k++)
    {
        a1 = A[r*n*16+k*16]; a2 = A[r*n*16+k*16+1]; a3 = A[r*n*16+k*16+2]; a4 = A[r*n*16+k*16+3]; a5 = A[r*n*16+k*16+4]; a6 = A[r*n*16+k*16+5]; a7 = A[r*n*16+k*16+6]; a8 = A[r*n*16+k*16+7]; a9 = A[r*n*16+k*16+8]; a10 = A[r*n*16+k*16+9]; a11 = A[r*n*16+k*16+10]; a12 = A[r*n*16+k*16+11]; a13 = A[r*n*16+k*16+12]; a14 = A[r*n*16+k*16+13]; a15 = A[r*n*16+k*16+14]; a16 = A[r*n*16+k*16+15];
        b1 = B[k*n*16+c*16]; b2 = B[k*n*16+c*16+1]; b3 = B[k*n*16+c*16+2]; b4 = B[k*n*16+c*16+3]; b5 = B[k*n*16+c*16+4]; b6 = B[k*n*16+c*16+5]; b7 = B[k*n*16+c*16+6]; b8 = B[k*n*16+c*16+7]; b9 = B[k*n*16+c*16+8]; b10 = B[k*n*16+c*16+9]; b11 = B[k*n*16+c*16+10]; b12 = B[k*n*16+c*16+11]; b13 = B[k*n*16+c*16+12]; b14 = B[k*n*16+c*16+13]; b15 = B[k*n*16+c*16+14]; b16 = B[k*n*16+c*16+15];

        hdf_mul(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,&c1,&c2,&c3,&c4,&c5,&c6,&c7,&c8,&c9,&c10,&c11,&c12,&c13,&c14,&c15,&c16);
        hdf_add(prd1,prd2,prd3,prd4,prd5,prd6,prd7,prd8,prd9,prd10,prd11,prd12,prd13,prd14,prd15,prd16,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,&prd1,&prd2,&prd3,&prd4,&prd5,&prd6,&prd7,&prd8,&prd9,&prd10,&prd11,&prd12,&prd13,&prd14,&prd15,&prd16);
    }
    C[r*n*16+c*16] = prd1;
    C[r*n*16+c*16+1] = prd2;
    C[r*n*16+c*16+2] = prd3;
    C[r*n*16+c*16+3] = prd4;
    C[r*n*16+c*16+4] = prd5;
    C[r*n*16+c*16+5] = prd6;
    C[r*n*16+c*16+6] = prd7;
    C[r*n*16+c*16+7] = prd8;
    C[r*n*16+c*16+8] = prd9;
    C[r*n*16+c*16+9] = prd10;
    C[r*n*16+c*16+10] = prd11;
    C[r*n*16+c*16+11] = prd12;
    C[r*n*16+c*16+12] = prd13;
    C[r*n*16+c*16+13] = prd14;
    C[r*n*16+c*16+14] = prd15;
    C[r*n*16+c*16+15] = prd16;
}

vector<double> matmulTCnt(vector<double> A,vector<double> B, int n, int nfrag, int p, float &t_CUDA) {
    vector<double> C = zeros(n,n,p);
    
    double* A_d;
    double* B_d;
    double* C_d;

    cudaMalloc((void**)&A_d,n*n*p*sizeof(double));
    cudaMemcpy(A_d,A.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d,n*n*p*sizeof(double));
    cudaMemcpy(B_d,B.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d,n*n*p*sizeof(double));
    cudaMemcpy(C_d,C.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);

    int nlen = n/nfrag;
    dim3 Gr(nlen,nlen);
    dim3 Bl(nfrag,nfrag);

    cudaEvent_t t0,t1;           // to measure time spent by kernels
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    if (p==2) {
	cudaEventRecord(t0);
        ddmm<<<Gr,Bl>>>(A_d,B_d,C_d,n);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
    } else if (p==4) {
	cudaEventRecord(t0);
	qdmm<<<Gr,Bl>>>(A_d,B_d,C_d,n);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
    } else if (p==8) {
	cudaEventRecord(t0);
	odmm<<<Gr,Bl>>>(A_d,B_d,C_d,n);
        cudaEventRecord(t1);
	cudaEventSynchronize(t1);
    } else if (p==16) {
	cudaEventRecord(t0);
	hdmm<<<Gr,Bl>>>(A_d,B_d,C_d,n);
	cudaEventRecord(t1);
	cudaEventSynchronize(t1);
    }
    cudaEventElapsedTime(&t_CUDA,t0,t1);
    cudaMemcpy(C.data(),C_d,n*n*p*sizeof(double),cudaMemcpyDeviceToHost);

    return C;
}

vector<double> matmulhost(vector<double> A,vector<double> B, int n, int p) {
    vector<double> C = zeros(n,n,p);
    for (int r=0;r<n;r++) {
	for (int c=0;c<n;c++) {
            for (int k=0;k<n;k++) {
		for (int i=0;i<p;i++) {
		    for (int j=0;j<=i;j++) {
			C[r*n*p+c*p+i] += A[r*n*p+k*p+j]*B[k*n*p+c*p+(i-j)];
		    }
		}	
	    }
	}
    }
    return C;
}

vector<double> matmulddf(vector<double> A,vector<double> B, int n) {
    vector<double> C = zeros(n,n,2);
    for (int r=0;r<n;r++) {
        for (int c=0;c<n;c++) {
            for (int k=0;k<n;k++) {
                double c_hi,c_lo;
		ddf_mul(A[r*n*2+k*2],A[r*n*2+k*2+1],B[k*n*2+c*2],B[k*n*2+c*2+1],&c_hi,&c_lo);
		ddf_add(C[r*n*2+c*2],C[r*n*2+c*2+1],c_hi,c_lo,&C[r*n*2+c*2],&C[r*n*2+c*2+1]);
            }
        }
    }
    return C;
}

vector<double> matmulqdf(vector<double> A,vector<double> B, int n) {
    vector<double> C = zeros(n,n,4);
    for (int r=0;r<n;r++) {
        for (int c=0;c<n;c++) {
            for (int k=0;k<n;k++) {
                double c1,c2,c3,c4;
                qdf_mul(A[r*n*4+k*4],A[r*n*4+k*4+1],A[r*n*4+k*4+2],A[r*n*4+k*4+3],B[k*n*4+c*4],B[k*n*4+c*4+1],B[k*n*4+c*4+2],B[k*n*4+c*4+3],&c1,&c2,&c3,&c4);
                qdf_add(C[r*n*4+c*4],C[r*n*4+c*4+1],C[r*n*4+c*4+2],C[r*n*4+c*4+3],c1,c2,c3,c4,&C[r*n*4+c*4],&C[r*n*4+c*4+1],&C[r*n*4+c*4+2],&C[r*n*4+c*4+3]);
            }
        }
    }
    return C;
}

vector<double> matmulodf(vector<double> A,vector<double> B, int n) {
    vector<double> C = zeros(n,n,8);
    for (int r=0;r<n;r++) {
        for (int c=0;c<n;c++) {
            for (int k=0;k<n;k++) {
                double c1,c2,c3,c4,c5,c6,c7,c8;
                odf_mul(A[r*n*8+k*8],A[r*n*8+k*8+1],A[r*n*8+k*8+2],A[r*n*8+k*8+3],A[r*n*8+k*8+4],A[r*n*8+k*8+5],A[r*n*8+k*8+6],A[r*n*8+k*8+7],B[k*n*8+c*8],B[k*n*8+c*8+1],B[k*n*8+c*8+2],B[k*n*8+c*8+3],B[k*n*8+c*8+4],B[k*n*8+c*8+5],B[k*n*8+c*8+6],B[k*n*8+c*8+7],&c1,&c2,&c3,&c4,&c5,&c6,&c7,&c8);
                odf_add(C[r*n*8+c*8],C[r*n*8+c*8+1],C[r*n*8+c*8+2],C[r*n*8+c*8+3],C[r*n*8+c*8+4],C[r*n*8+c*8+5],C[r*n*8+c*8+6],C[r*n*8+c*8+7],c1,c2,c3,c4,c5,c6,c7,c8,&C[r*n*8+c*8],&C[r*n*8+c*8+1],&C[r*n*8+c*8+2],&C[r*n*8+c*8+3],&C[r*n*8+c*8+4],&C[r*n*8+c*8+5],&C[r*n*8+c*8+6],&C[r*n*8+c*8+7]);
            }                                                                                                                   }
    }
    return C;
}

vector<double> matmulhdf(vector<double> A,vector<double> B, int n) {
    vector<double> C = zeros(n,n,16);
    for (int r=0;r<n;r++) {
	for (int c=0;c<n;c++) {
            for (int k=0;k<n;k++) {
       	        double c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16;
		hdf_mul(A[r*n*16+k*16],A[r*n*16+k*16+1],A[r*n*16+k*16+2],A[r*n*16+k*16+3],A[r*n*16+k*16+4],A[r*n*16+k*16+5],A[r*n*16+k*16+6],A[r*n*16+k*16+7],A[r*n*16+k*16+8],A[r*n*16+k*16+9],A[r*n*16+k*16+10],A[r*n*16+k*16+11],A[r*n*16+k*16+12],A[r*n*16+k*16+13],A[r*n*16+k*16+14],A[r*n*16+k*16+15],B[k*n*16+c*16],B[k*n*16+c*16+1],B[k*n*16+c*16+2],B[k*n*16+c*16+3],B[k*n*16+c*16+4],B[k*n*16+c*16+5],B[k*n*16+c*16+6],B[k*n*16+c*16+7],B[k*n*16+c*16+8],B[k*n*16+c*16+9],B[k*n*16+c*16+10],B[k*n*16+c*16+11],B[k*n*16+c*16+12],B[k*n*16+c*16+13],B[k*n*16+c*16+14],B[k*n*16+c*16+15],&c1,&c2,&c3,&c4,&c5,&c6,&c7,&c8,&c9,&c10,&c11,&c12,&c13,&c14,&c15,&c16);
                hdf_add(C[r*n*16+c*16],C[r*n*16+c*16+1],C[r*n*16+c*16+2],C[r*n*16+c*16+3],C[r*n*16+c*16+4],C[r*n*16+c*16+5],C[r*n*16+c*16+6],C[r*n*16+c*16+7],C[r*n*16+c*16+8],C[r*n*16+c*16+9],C[r*n*16+c*16+10],C[r*n*16+c*16+11],C[r*n*16+c*16+12],C[r*n*16+c*16+13],C[r*n*16+c*16+14],C[r*n*16+c*16+15],c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,&C[r*n*16+c*16],&C[r*n*16+c*16+1],&C[r*n*16+c*16+2],&C[r*n*16+c*16+3],&C[r*n*16+c*16+4],&C[r*n*16+c*16+5],&C[r*n*16+c*16+6],&C[r*n*16+c*16+7],&C[r*n*16+c*16+8],&C[r*n*16+c*16+9],&C[r*n*16+c*16+10],&C[r*n*16+c*16+11],&C[r*n*16+c*16+12],&C[r*n*16+c*16+13],&C[r*n*16+c*16+14],&C[r*n*16+c*16+15]);
            }                                                                                                                   }
    }
    return C;
}

void renormhost(vector<double> &A,int n,int p) {
    for (int i=0;i<n;i++) {
	for (int j=0;j<n;j++) {
	    for (int k=0;k<p-1;k++) {
		double newhi = A[i*n*p+j*p+k]+A[i*n*p+j*p+(k+1)];
		double newlo = newhi-A[i*n*p+j*p+k];
		A[i*n*p+j*p+k] = newhi;
		A[i*n*p+j*p+(k+1)] = newlo;
	    }
	}
    }
}

void renormhostbig(vector<double> &A,int n,int p) {
    for (int i=0;i<n;i++) {
        for (int j=0;j<n;j++) {
            for (int k=0;k<p-1;k++) {
                double newhi = A[i*n*p+k*n+j]+A[i*n*p+(k+1)*n+j];
                double newlo = newhi-A[i*n*p+k*n+j];
                A[i*n*p+k*n+j] = newhi;
                A[i*n*p+(k+1)*n+j] = newlo;
            }
        }
    }
}

__global__ void pllsqueeze_kernel_2_8(double *x1,double *x2,double *x3,double *x4,double *x5,double *x6,double *x7,double *x8,double *y1,double *y2) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    ddf_inc_d(&y1[i],&y2[i],x1[i]);
    ddf_inc_d(&y1[i],&y2[i],x2[i]);
    ddf_inc_d(&y1[i],&y2[i],x3[i]);
    ddf_inc_d(&y1[i],&y2[i],x4[i]);
    ddf_inc_d(&y1[i],&y2[i],x5[i]);
    ddf_inc_d(&y1[i],&y2[i],x6[i]);
    ddf_inc_d(&y1[i],&y2[i],x7[i]);
    ddf_inc_d(&y1[i],&y2[i],x8[i]);
}

vector<double> pllsqueeze_2_8(vector<double> x) {
    int n = x.size()/8;
    vector<double> X1 = part(x,8,0);
    vector<double> X2 = part(x,8,1);
    vector<double> X3 = part(x,8,2);
    vector<double> X4 = part(x,8,3);
    vector<double> X5 = part(x,8,4);
    vector<double> X6 = part(x,8,5);
    vector<double> X7 = part(x,8,6);
    vector<double> X8 = part(x,8,7);
    vector<double> Y1 = zeros(n,1,1);
    vector<double> Y2 = zeros(n,1,1);
    double* x1;
    double* x2;
    double* x3;
    double* x4;
    double* x5;
    double* x6;
    double* x7;
    double* x8;
    double* y1;
    double* y2;
    cudaMalloc((void**)&x1,n*sizeof(double));
    cudaMemcpy(x1,X1.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x2,n*sizeof(double));
    cudaMemcpy(x2,X2.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x3,n*sizeof(double));
    cudaMemcpy(x3,X3.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x4,n*sizeof(double));
    cudaMemcpy(x4,X4.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x5,n*sizeof(double));
    cudaMemcpy(x5,X5.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x6,n*sizeof(double));
    cudaMemcpy(x6,X6.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x7,n*sizeof(double));
    cudaMemcpy(x7,X7.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x8,n*sizeof(double));
    cudaMemcpy(x8,X8.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&y1,n*sizeof(double));
    cudaMemcpy(y1,Y1.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&y2,n*sizeof(double));
    cudaMemcpy(y2,Y2.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    dim3 G;
    dim3 B;
    if (n <64) {
        G.x = 1;
        G.y = 1;
        B.x = n;
        B.y = 1;
    } else {
        G.x = n/64; // assume 64|n
        G.y = 1;
        B.x = 64;
        B.y = 1;
    }
    pllsqueeze_kernel_2_8<<<G,B>>>(x1,x2,x3,x4,x5,x6,x7,x8,y1,y2);
    cudaMemcpy(Y1.data(),y1,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Y2.data(),y2,n*sizeof(double),cudaMemcpyDeviceToHost);
    vector<double> y;
    for (int i=0;i<n;i++) {
        y.push_back(Y1[i]);
        y.push_back(Y2[i]);
    }
    return y;
}

vector<double> pllntsqueeze_2_8(vector<double> x) {
    int n = x.size()/8;
    vector<double> X1 = part(x,8,0);
    vector<double> X2 = part(x,8,1);
    vector<double> X3 = part(x,8,2);
    vector<double> X4 = part(x,8,3);
    vector<double> X5 = part(x,8,4);
    vector<double> X6 = part(x,8,5);
    vector<double> X7 = part(x,8,6);
    vector<double> X8 = part(x,8,7);
    vector<double> Y1 = zeros(n,1,1);
    vector<double> Y2 = zeros(n,1,1);
    for (int i=0;i<n;i++) {
        ddf_inc_d(&Y1[i],&Y2[i],X1[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X2[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X3[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X4[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X5[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X6[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X7[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X8[i]);
    }
    vector<double> y;
    for (int i=0;i<n;i++) {
        y.push_back(Y1[i]);
        y.push_back(Y2[i]);
    }
    return y;
}

__global__ void pllsqueeze_kernel_2_12(double *x1,double *x2,double *x3,double *x4,double *x5,double *x6,double *x7,double *x8,double *x9,double *x10,double *x11,double *x12,double *y1,double *y2) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    ddf_inc_d(&y1[i],&y2[i],x1[i]);
    ddf_inc_d(&y1[i],&y2[i],x2[i]);
    ddf_inc_d(&y1[i],&y2[i],x3[i]);
    ddf_inc_d(&y1[i],&y2[i],x4[i]);
    ddf_inc_d(&y1[i],&y2[i],x5[i]);
    ddf_inc_d(&y1[i],&y2[i],x6[i]);
    ddf_inc_d(&y1[i],&y2[i],x7[i]);
    ddf_inc_d(&y1[i],&y2[i],x8[i]);
    ddf_inc_d(&y1[i],&y2[i],x9[i]);
    ddf_inc_d(&y1[i],&y2[i],x10[i]);
    ddf_inc_d(&y1[i],&y2[i],x11[i]);
    ddf_inc_d(&y1[i],&y2[i],x12[i]);
}

vector<double> pllsqueeze_2_12(vector<double> x) {
    int n = x.size()/12;
    vector<double> X1 = part(x,12,0);
    vector<double> X2 = part(x,12,1);
    vector<double> X3 = part(x,12,2);
    vector<double> X4 = part(x,12,3);
    vector<double> X5 = part(x,12,4);
    vector<double> X6 = part(x,12,5);
    vector<double> X7 = part(x,12,6);
    vector<double> X8 = part(x,12,7);
    vector<double> X9 = part(x,12,8);
    vector<double> X10 = part(x,12,9);
    vector<double> X11 = part(x,12,10);
    vector<double> X12 = part(x,12,11);
    vector<double> Y1 = zeros(n,1,1);
    vector<double> Y2 = zeros(n,1,1);
    
    double* x1;
    double* x2;
    double* x3;
    double* x4;
    double* x5;
    double* x6;
    double* x7;
    double* x8;
    double* x9;
    double* x10;
    double* x11;
    double* x12;
    double* y1;
    double* y2;
    cudaMalloc((void**)&x1,n*sizeof(double));
    cudaMemcpy(x1,X1.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x2,n*sizeof(double));
    cudaMemcpy(x2,X2.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x3,n*sizeof(double));
    cudaMemcpy(x3,X3.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x4,n*sizeof(double));
    cudaMemcpy(x4,X4.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x5,n*sizeof(double));
    cudaMemcpy(x5,X5.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x6,n*sizeof(double));
    cudaMemcpy(x6,X6.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x7,n*sizeof(double));
    cudaMemcpy(x7,X7.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x8,n*sizeof(double));
    cudaMemcpy(x8,X8.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x9,n*sizeof(double));
    cudaMemcpy(x9,X9.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x10,n*sizeof(double));
    cudaMemcpy(x10,X10.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x11,n*sizeof(double));
    cudaMemcpy(x11,X11.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x12,n*sizeof(double));
    cudaMemcpy(x12,X12.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&y1,n*sizeof(double));
    cudaMemcpy(y1,Y1.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&y2,n*sizeof(double));
    cudaMemcpy(y2,Y2.data(),n*sizeof(double),cudaMemcpyHostToDevice);

    dim3 G;
    dim3 B;
    if (n <64) {
        G.x = 1;
        G.y = 1;
        B.x = n;
        B.y = 1;
    } else {
        G.x = n/64; // assume 64|n
        G.y = 1;
        B.x = 64;
        B.y = 1;
    }
    pllsqueeze_kernel_2_12<<<G,B>>>(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y1,y2);
    cudaMemcpy(Y1.data(),y1,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Y2.data(),y2,n*sizeof(double),cudaMemcpyDeviceToHost);
    vector<double> y;
    for (int i=0;i<n;i++) {
	y.push_back(Y1[i]);
	y.push_back(Y2[i]);
    }
    return y;
}

vector<double> pllntsqueeze_2_12(vector<double> x) {
    int n = x.size()/12;
    vector<double> X1 = part(x,12,0);
    vector<double> X2 = part(x,12,1);
    vector<double> X3 = part(x,12,2);
    vector<double> X4 = part(x,12,3);
    vector<double> X5 = part(x,12,4);
    vector<double> X6 = part(x,12,5);
    vector<double> X7 = part(x,12,6);
    vector<double> X8 = part(x,12,7);
    vector<double> X9 = part(x,12,8);
    vector<double> X10 = part(x,12,9);
    vector<double> X11 = part(x,12,10);
    vector<double> X12 = part(x,12,11);
    vector<double> Y1 = zeros(n,1,1);
    vector<double> Y2 = zeros(n,1,1);
    for (int i=0;i<n;i++) {
        ddf_inc_d(&Y1[i],&Y2[i],X1[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X2[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X3[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X4[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X5[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X6[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X7[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X8[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X9[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X10[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X11[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X12[i]);
    }
    vector<double> y;
    for (int i=0;i<n;i++) {
        y.push_back(Y1[i]);
        y.push_back(Y2[i]);
    }
    return y;
}

__global__ void pllsqueeze_kernel_2_16(double *x1,double *x2,double *x3,double *x4,double *x5,double *x6,double *x7,double *x8,double *x9,double *x10,double *x11,double *x12,double *x13,double *x14,double *x15,double *x16,double *y1,double *y2) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    ddf_inc_d(&y1[i],&y2[i],x1[i]);
    ddf_inc_d(&y1[i],&y2[i],x2[i]);
    ddf_inc_d(&y1[i],&y2[i],x3[i]);
    ddf_inc_d(&y1[i],&y2[i],x4[i]);
    ddf_inc_d(&y1[i],&y2[i],x5[i]);
    ddf_inc_d(&y1[i],&y2[i],x6[i]);
    ddf_inc_d(&y1[i],&y2[i],x7[i]);
    ddf_inc_d(&y1[i],&y2[i],x8[i]);
    ddf_inc_d(&y1[i],&y2[i],x9[i]);
    ddf_inc_d(&y1[i],&y2[i],x10[i]);
    ddf_inc_d(&y1[i],&y2[i],x11[i]);
    ddf_inc_d(&y1[i],&y2[i],x12[i]);
    ddf_inc_d(&y1[i],&y2[i],x13[i]);
    ddf_inc_d(&y1[i],&y2[i],x14[i]);
    ddf_inc_d(&y1[i],&y2[i],x15[i]);
    ddf_inc_d(&y1[i],&y2[i],x16[i]);
}

vector<double> pllsqueeze_2_16(vector<double> x) {
    int n = x.size()/16;
    vector<double> X1 = part(x,16,0);
    vector<double> X2 = part(x,16,1);
    vector<double> X3 = part(x,16,2);
    vector<double> X4 = part(x,16,3);
    vector<double> X5 = part(x,16,4);
    vector<double> X6 = part(x,16,5);
    vector<double> X7 = part(x,16,6);
    vector<double> X8 = part(x,16,7);
    vector<double> X9 = part(x,16,8);
    vector<double> X10 = part(x,16,9);
    vector<double> X11 = part(x,16,10);
    vector<double> X12 = part(x,16,11);
    vector<double> X13 = part(x,16,12);
    vector<double> X14 = part(x,16,13);
    vector<double> X15 = part(x,16,14);
    vector<double> X16 = part(x,16,15);
    vector<double> Y1 = zeros(n,1,1);
    vector<double> Y2 = zeros(n,1,1);
    double* x1;
    double* x2;
    double* x3;
    double* x4;
    double* x5;
    double* x6;
    double* x7;
    double* x8;
    double* x9;
    double* x10;
    double* x11;
    double* x12;
    double* x13;
    double* x14;
    double* x15;
    double* x16;
    double* y1;
    double* y2;
    cudaMalloc((void**)&x1,n*sizeof(double));
    cudaMemcpy(x1,X1.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x2,n*sizeof(double));
    cudaMemcpy(x2,X2.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x3,n*sizeof(double));
    cudaMemcpy(x3,X3.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x4,n*sizeof(double));
    cudaMemcpy(x4,X4.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x5,n*sizeof(double));
    cudaMemcpy(x5,X5.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x6,n*sizeof(double));
    cudaMemcpy(x6,X6.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x7,n*sizeof(double));
    cudaMemcpy(x7,X7.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x8,n*sizeof(double));
    cudaMemcpy(x8,X8.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x9,n*sizeof(double));
    cudaMemcpy(x9,X9.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x10,n*sizeof(double));
    cudaMemcpy(x10,X10.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x11,n*sizeof(double));
    cudaMemcpy(x11,X11.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x12,n*sizeof(double));
    cudaMemcpy(x12,X12.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x13,n*sizeof(double));
    cudaMemcpy(x13,X13.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x14,n*sizeof(double));
    cudaMemcpy(x14,X14.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x15,n*sizeof(double));
    cudaMemcpy(x15,X15.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&x16,n*sizeof(double));
    cudaMemcpy(x16,X16.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&y1,n*sizeof(double));
    cudaMemcpy(y1,Y1.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&y2,n*sizeof(double));
    cudaMemcpy(y2,Y2.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    dim3 G;
    dim3 B;
    if (n <64) {
        G.x = 1;
        G.y = 1;
        B.x = n;
        B.y = 1;
    } else {
        G.x = n/64; // assume 64|n
        G.y = 1;
        B.x = 64;
        B.y = 1;
    }
    pllsqueeze_kernel_2_16<<<G,B>>>(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,y1,y2);
    cudaMemcpy(Y1.data(),y1,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Y2.data(),y2,n*sizeof(double),cudaMemcpyDeviceToHost);
    vector<double> y;
    for (int i=0;i<n;i++) {
        y.push_back(Y1[i]);
        y.push_back(Y2[i]);
    }
    return y;
}

vector<double> pllntsqueeze_2_16(vector<double> x) {
    int n = x.size()/16;
    vector<double> X1 = part(x,16,0);
    vector<double> X2 = part(x,16,1);
    vector<double> X3 = part(x,16,2);
    vector<double> X4 = part(x,16,3);
    vector<double> X5 = part(x,16,4);
    vector<double> X6 = part(x,16,5);
    vector<double> X7 = part(x,16,6);
    vector<double> X8 = part(x,16,7);
    vector<double> X9 = part(x,16,8);
    vector<double> X10 = part(x,16,9);
    vector<double> X11 = part(x,16,10);
    vector<double> X12 = part(x,16,11);
    vector<double> X13 = part(x,16,12);
    vector<double> X14 = part(x,16,13);
    vector<double> X15 = part(x,16,14);
    vector<double> X16 = part(x,16,15);
    vector<double> Y1 = zeros(n,1,1);
    vector<double> Y2 = zeros(n,1,1);
    for (int i=0;i<n;i++) {
        ddf_inc_d(&Y1[i],&Y2[i],X1[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X2[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X3[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X4[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X5[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X6[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X7[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X8[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X9[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X10[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X11[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X12[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X13[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X14[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X15[i]);
        ddf_inc_d(&Y1[i],&Y2[i],X16[i]);
    }
    vector<double> y;
    for (int i=0;i<n;i++) {
        y.push_back(Y1[i]);
        y.push_back(Y2[i]);
    }
    return y;
}

vector<double> pllsqueeze(vector<double> x,int p,int pp) {
    if (pp==12) {
	return pllsqueeze_2_12(x);
    } else if (p==2) {
	if (pp==8) {
            return pllsqueeze_2_8(x);
	} else if (pp==16) {
	    return pllsqueeze_2_16(x);
	}
    }
}
    /*
    } else if (p==4) {
	if (pp=8) {
            return pllsqueeze_4_8(x);
        } else if (pp=16) {
            return pllsqueeze_4_16(x);
        }
    } else if (p==8) {
	if (pp=8) {
            return pllsqueeze_8_8(x);
        } else if (pp=16) {
            return pllsqueeze_8_16(x);
        }
    } else if (p==16) {
	if (pp=8) {
            return pllsqueeze_16_8(x);
        } else if (pp=16) {
            return pllsqueeze_16_16(x);
        }
    }
    */

vector<double> pllntsqueeze(vector<double> x,int p,int pp) {
    if (pp==12) {
        return pllntsqueeze_2_12(x);
    } else if (p==2) {
        if (pp==8) {
            return pllntsqueeze_2_8(x);
        } else if (pp==16) {
            return pllntsqueeze_2_16(x);
        }
    }
}
    /*
    } else if (p==4) {
        if (pp=8) {
            return pllntsqueeze_4_8(x);
        } else if (pp=16) {
            return pllntsqueeze_4_16(x);
        }
    } else if (p==8) {
        if (pp=8) {
            return pllntsqueeze_8_8(x);
        } else if (pp=16) {
            return pllntsqueeze_8_16(x);
        }
    } else if (p==16) {
        if (pp=8) {
            return pllntsqueeze_16_8(x);
        } else if (pp=16) {
            return pllntsqueeze_16_16(x);
        }
    }
    */

__global__ void pllsqueeze_2_pp_old(double *x,double *y,int pp) {
   int i=blockDim.x*blockIdx.x+threadIdx.x;
   for (int j=0;j<pp;j++) {
       ddf_inc_d(&y[2*i],&y[2*i+1],x[pp*i+j]);
       __syncthreads();
   }
}

__global__ void pllsqueeze_4_pp_old(double *x,double *y,int pp) {
   int i=blockDim.x*blockIdx.x+threadIdx.x;
   for (int j=0;j<pp;j++) {
       qdf_inc_d(&y[4*i],&y[4*i+1],&y[4*i+2],&y[4*i+3],x[pp*i+j]);
       __syncthreads();
   }
}

__global__ void pllsqueeze_8_pp_old(double *x,double *y,int pp) {
   int i=blockDim.x*blockIdx.x+threadIdx.x;
   for (int j=0;j<pp;j++) {
       odf_inc_d(&y[8*i],&y[8*i+1],&y[8*i+2],&y[8*i+3],&y[8*i+4],&y[8*i+5],&y[8*i+6],&y[8*i+7],x[pp*i+j]);
       __syncthreads();
   }
}

__global__ void pllsqueeze_16_pp_old(double *x,double *y,int pp) {
   int i=blockDim.x*blockIdx.x+threadIdx.x;
   for (int j=0;j<pp;j++) {
       hdf_inc_d(&y[16*i],&y[16*i+1],&y[16*i+2],&y[16*i+3],&y[16*i+4],&y[16*i+5],&y[16*i+6],&y[16*i+7],&y[16*i+8],&y[16*i+9],&y[16*i+10],&y[16*i+11],&y[16*i+12],&y[16*i+13],&y[16*i+14],&y[16*i+15],x[pp*i+j]);
       __syncthreads();
   }
}

vector<double> pllsqueeze_old(vector<double> x,int p,int pp) {
    int n = x.size()/pp;
    vector<double> y = zeros(n,1,p);

    double* x_d;
    double* y_d;
    cudaMalloc((void**)&x_d,n*pp*sizeof(double));
    cudaMemcpy(x_d,x.data(),n*pp*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&y_d,n*p*sizeof(double));
    cudaMemcpy(y_d,y.data(),n*p*sizeof(double),cudaMemcpyHostToDevice);
    dim3 G;
    dim3 B;
    if (n <64) {
        G.x = 1;
        G.y = 1;
        B.x = n;
        B.y = 1;
    } else {
        G.x = n/64; // assume 64|n
        G.y = 1;
        B.x = 64;
        B.y = 1;
    }
    if (p==2) {
	pllsqueeze_2_pp_old<<<G,B>>>(x_d,y_d,pp);
    } else if (p==4) {
	pllsqueeze_4_pp_old<<<G,B>>>(x_d,y_d,pp);
    } else if (p==8) {
        pllsqueeze_8_pp_old<<<G,B>>>(x_d,y_d,pp);
    } else if (p==16) {
        pllsqueeze_16_pp_old<<<G,B>>>(x_d,y_d,pp);
    }
    cudaMemcpy(y.data(),y_d,n*p*sizeof(double),cudaMemcpyDeviceToHost);
    return y;
}


/*
   Python code to generate instructions for pllsqueeze_kernel_p_pp, pllsqueeze_p_pp, and pllntsqueeze_p_pp

def pllsqueeze_kernel(p,pp):
    code = f"__global__ void pllsqueeze_kernel_{p}_{pp}("+",".join([f"double *x{j+1}" for j in range(pp)]+[f"double *y{i+1}" for i in range(p)])+") {\n"
    code += "    int i=blockDim.x*blockIdx.x+threadIdx.x;\n"
    if p==2:
        mode = "ddf"
    elif p==4:
        mode = "qdf"
    elif p==8:
        mode = "odf"
    elif p==16:
        mode = "hdf"
    for j in range(pp):
        code += f"    {mode}_inc_d("+",".join([f"&y{l+1}[i]" for l in range(p)])+f",x{j+1}[i]);\n"
    return code+"}"

def pllsqueeze(p,pp):
    code = f"vector<double> pllsqueeze_{p}_{pp}(vector<double> x)"+" {\n"
    code += f"    int n = x.size()/{pp};\n"
    for j in range(pp):
        code += f"    vector<double> X{j+1} = part(x,{pp},{j});\n"
    for i in range(p):
        code += f"    vector<double> Y{i+1} = zeros(n,1,1);\n"
    code += "\n".join([f"double *x{j+1};" for j in range(pp)]+[f"double* y{i+1};" for i in range(p)])+";\n"
    for j in range(pp):
        code += f"    cudaMalloc((void**)&x{j+1},n*sizeof(double));\n"
        code += f"    cudaMemcpy(x{j+1},X{j+1}.data(),n*sizeof(double),cudaMemcpyHostToDevice);\n"
    for i in range(p):
        code += f"    cudaMalloc((void**)&y{i+1},n*sizeof(double));\n"
        code += f"    cudaMemcpy(y{i+1},Y{i+1}.data(),n*sizeof(double),cudaMemcpyHostToDevice);\n"
    code += "    dim3 G;\n    dim3 B;\n    if (n <64) {\n        G.x = 1;\n        G.y = 1;\n        B.x = n;\n        B.y = 1;\n    } else {\n        G.x = n/64; // assume 64|n\n        G.y = 1;\n        B.x = 64;\n        B.y = 1;\n    }"
    code += f"    pllsqueeze_{p}_{pp}<<<G,B>>>("+",".join([f"x{j+1}" for j in range(pp)]+[f"y{i+1}" for i in range(p)])+");\n"
    for i in range(p):
        code += f"    cudaMemcpy(Y{i+1}.data(),y{i+1},n*sizeof(double),cudaMemcpyDeviceToHost);\n"
    code += "    vector<double> y;\n    for (int i=0;i<n;i++) {\n"
    for j in range(p):
        code += f"        y.push_back(Y{j+1}[i]);\n"
    return code + "    }\n    return y;\n}"


def pllntsqueeze(p,pp):
    code = f"vector<double> pllntsqueeze_{p}_{pp}(vector<double> x)"+" {\n"
    code += f"    int n = x.size()/{pp};\n"
    for j in range(pp):
        code += f"    vector<double> X{j+1} = part(x,{pp},{j});\n"
    for i in range(p):
        code += f"    vector<double> Y{i+1} = zeros(n,1,1);\n"
    if p==2:
        mode = "ddf"
    elif p==4:
        mode = "qdf"
    elif p==8:
        mode = "odf"
    elif p==16:
        mode = "hdf"
    code += "    for (int i=0;i<n;i++) {\n"
    for j in range(pp):
        code += f"        {mode}_inc_d("+",".join([f"&Y{l+1}[i]" for l in range(p)])+f",X{j+1}[i]);\n"
    code += "    }\n    vector<double> y;\n    for (int i=0;i<n;i++) {\n"
    for j in range(p):
        code += f"        y.push_back(Y{j+1}[i]);\n"
    return code + "    }\n    return y;\n}"
*/

