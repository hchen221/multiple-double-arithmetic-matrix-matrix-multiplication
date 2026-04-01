#include "TCcuda.h"
#include "floatybits.h"
#include <stdio.h>
#include <iostream>
#include "double_double_functions.cu"
#include "quad_double_functions.cu"
//#include "penta_double_functions.cu"
#include "hexa_double_functions.cu"
#include "octo_double_functions.cu"
//#include "deca_double_functions.cu"

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

__global__ void matmulkernel(double* A,double* B, double* C,int n, int p) {
    int r = blockIdx.x;
    int c = blockIdx.y;
    int i = threadIdx.x;
    for (int k=0;k<n;k++) {
        for (int j=0;j<p;j++) {
	    if (j<=i) {
		C[r*n*p+c*p+i] += A[r*n*p+k*p+j]*B[k*n*p+c*p+(i-j)];
	    } else {
	        C[r*n*p+c*p+i] += 0*0;
	    }
	    __syncthreads();
	}
    }
}

vector<double> matmulTCnt(vector<double> A,vector<double> B, int n, int p) {
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

    dim3 Gr(n,n);
    dim3 Bl(p,1);
    matmulkernel<<<Gr,Bl>>>(A_d,B_d,C_d,n,p);
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
/*
vector<double> matmulpdf(vector<double> A,vector<double> B, int n) {
    vector<double> C = zeros(n,n,5);
    for (int r=0;r<n;r++) {
        for (int c=0;c<n;c++) {
            for (int k=0;k<n;k++) {
                double c1,c2,c3,c4,c5;
                pdf_mul(A[r*n*5+k*5],A[r*n*5+k*5+1],A[r*n*5+k*5+2],A[r*n*5+k*5+3],A[r*n*5+k*5+4],B[k*n*5+c*5],B[k*n*5+c*5+1],B[k*n*5+c*5+2],B[k*n*5+c*5+3],B[k*n*5+c*5+4],&c1,&c2,&c3,&c4,&c5);
                pdf_add(C[r*n*5+c*5],C[r*n*5+c*5+1],C[r*n*5+c*5+2],C[r*n*5+c*5+3],C[r*n*5+c*5+4],c1,c2,c3,c4,c5,&C[r*n*5+c*5],&C[r*n*5+c*5+1],&C[r*n*5+c*5+2],&C[r*n*5+c*5+3],&C[r*n*5+c*5+4]);
            }
        }
    }
    return C;
}
*/
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
/*
vector<double> matmuldaf(vector<double> A,vector<double> B, int n) {
    vector<double> C = zeros(n,n,10);
    for (int r=0;r<n;r++) {
        for (int c=0;c<n;c++) {
            for (int k=0;k<n;k++) {
                double c1,c2,c3,c4,c5,c6,c7,c8,c9,c10;
                daf_mul(A[r*n*10+k*10],A[r*n*10+k*10+1],A[r*n*10+k*10+2],A[r*n*10+k*10+3],A[r*n*10+k*10+4],A[r*n*10+k*10+5],A[r*n*10+k*10+6],A[r*n*10+k*10+7],A[r*n*10+k*10+8],A[r*n*10+k*10+9],B[k*n*10+c*10],B[k*n*10+c*10+1],B[k*n*10+c*10+2],B[k*n*10+c*10+3],B[k*n*10+c*10+4],B[k*n*10+c*10+5],B[k*n*10+c*10+6],B[k*n*10+c*10+7],B[k*n*10+c*10+8],B[k*n*10+c*10+9],&c1,&c2,&c3,&c4,&c5,&c6,&c7,&c8,&c9,&c10);
                daf_add(C[r*n*10+c*10],C[r*n*10+c*10+1],C[r*n*10+c*10+2],C[r*n*10+c*10+3],C[r*n*10+c*10+4],C[r*n*10+c*10+5],C[r*n*10+c*10+6],C[r*n*10+c*10+7],C[r*n*10+c*10+8],C[r*n*10+c*10+9],c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,&C[r*n*10+c*10],&C[r*n*10+c*10+1],&C[r*n*10+c*10+2],&C[r*n*10+c*10+3],&C[r*n*10+c*10+4],&C[r*n*10+c*10+5],&C[r*n*10+c*10+6],&C[r*n*10+c*10+7],&C[r*n*10+c*10+8],&C[r*n*10+c*10+9]);
            }
        }
    }
    return C;
}
*/

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

vector<double> squeeze(vector<double> x,int p,int q) {
    int n = x.size()/(p*q);
    vector<double> y = zeros(n,1,p);
    for (int i=0;i<n;i++) {

        y[p*i] = x[p*q*i];
	for (int j=1;j<p*q;j++) {
		if (p==2) {
			ddf_inc_d(&y[p*i],&y[p*i+1],x[p*q*i+j]);
		} else if (p==4) {
			qdf_inc_d(&y[p*i],&y[p*i+1],&y[p*i+2],&y[p*i+3],x[p*q*i+j]);
		} else if (p==8) {
			odf_inc_d(&y[p*i],&y[p*i+1],&y[p*i+2],&y[p*i+3],&y[p*i+4],&y[p*i+5],&y[p*i+6],&y[p*i+7],x[p*q*i+j]);
		} else if (p==16) {
			hdf_inc_d(&y[p*i],&y[p*i+1],&y[p*i+2],&y[p*i+3],&y[p*i+4],&y[p*i+5],&y[p*i+6],&y[p*i+7],&y[p*i+8],&y[p*i+9],&y[p*i+10],&y[p*i+11],&y[p*i+12],&y[p*i+13],&y[p*i+14],&y[p*i+15],x[p*q*i+j]);
        }
    }

    }
    return y;
}

__global__ void pllsqueezekernel(double *x,double *y,int p,int q) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    y[p*i] = x[p*q*i];
    for (int j=1;j<p*q;j++) {
	if (p==2) {
            ddf_inc_d(&y[p*i],&y[p*i+1],x[p*q*i+j]);
	} else if (p==4) {
	    qdf_inc_d(&y[p*i],&y[p*i+1],&y[p*i+2],&y[p*i+3],x[p*q*i+j]);
	} else if (p==8) {
	    odf_inc_d(&y[p*i],&y[p*i+1],&y[p*i+2],&y[p*i+3],&y[p*i+4],&y[p*i+5],&y[p*i+6],&y[p*i+7],x[p*q*i+j]);
	} else if (p==16) {
	    hdf_inc_d(&y[p*i],&y[p*i+1],&y[p*i+2],&y[p*i+3],&y[p*i+4],&y[p*i+5],&y[p*i+6],&y[p*i+7],&y[p*i+8],&y[p*i+9],&y[p*i+10],&y[p*i+11],&y[p*i+12],&y[p*i+13],&y[p*i+14],&y[p*i+15],x[p*q*i+j]);
	}
	__syncthreads();
    }
}

vector<double> pllsqueeze(vector<double> x,int p,int q) {
    int n = x.size()/(p*q);
    vector<double> y = zeros(n,1,p);

    double* x_d;
    double* y_d;
    cudaMalloc((void**)&x_d,n*p*q*sizeof(double));
    cudaMemcpy(x_d,x.data(),n*p*q*sizeof(double),cudaMemcpyHostToDevice);
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

    pllsqueezekernel<<<G,B>>>(x_d,y_d,p,q);

    cudaMemcpy(y.data(),y_d,n*p*sizeof(double),cudaMemcpyDeviceToHost);

    return y;
}
