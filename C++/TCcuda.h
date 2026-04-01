#ifndef _TCFLAT_
#define _TCFLAT_

#include "floatybits.h"
#include "double_double_functions.h"
#include "quad_double_functions.h"
#include "octo_double_functions.h"
#include "hexa_double_functions.h"


#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;


/*
Assume matrices have a flat representation by default
mat(n,p) returns a flattened nxn matrix of random p doubles
*/
vector<double> mat(int n, int p, int expmin=0, int expmax=0);

/*
zeros(m,n,p) returns a matrix of all 0's
*/
vector<double> zeros(int m,int n, int p);

/*bigA(A,n,p) takes an nxn matrix of p-double entries A and returns [A_1,...,A_p] stacked row wise, formatted row major*/
vector<double> bigA(vector<double> A,int n,int p);

/*bigB(B,n,p) takes an nxn matrix of p-double entries B and returns the following
  [B_1,B_2,...,B_p]
  [0  ,B_1,...,B_{p-1}]
  [.  ,.  ,.  ..  ]
  [.  ,.  ,.  ,B_1]
  formatted col major*/
vector<double> bigB(vector<double> B,int n,int p);

/*bigB2(B,n,p) takes an nxn matrix of p-double entries B and returns an np x np matrix where each pxp block corresponds to an entry of B in the following form
  [b_1,b_2,...,b_p]
  [0  ,b_1,...,b_{p-1}]
  [.  ,.  ,.  ..  ]
  [.  ,.  ,.  ,b_1]
 formatter col major 
*/
vector<double> bigB2(vector<double> B,int n,int p);

/*renormbigA takes a matrix of form bigA(A,n,p) and runs on nxn threads in a 1x1 block to renormalize into non overlapping p-doubles component wise*/
__global__ void renormbigA(double* A,int n,int p);

/*renormA takes an nxn matrix A of p-doubles and runs on nxn threads in a 1x1 block to renormalize into non overlapping p-doubles component wise*/
__global__ void renormA(double* A,int n,int p);

/*matmulTCnt(A,B,n,p) takes nxn matrices A,B of p-doubles and computes the product C, done with regular CUDA cores, with helper functions ddmm,qdmm,odmm,hdmm depending on p, executed on nlen x nlen blocks with block size nfrag x nfrag where nfrag*nlen=n. The kernels are adapted from ddmm_kernels.cu in PHCpack*/
__global__ void ddmm(double *A,double *B,double *C,int n);
__global__ void qdmm(double *A,double *B,double *C,int n);
__global__ void odmm(double *A,double *B,double *C,int n);
__global__ void hdmm(double *A,double *B,double *C,int n);
vector<double> matmulTCnt(vector<double> A,vector<double> B,int n,int nfrag,int p);

/*matmulhost(A,B,n,p) takes nxn matrices A,B of p-doubles and computes the product C, all done on the host*/
vector<double> matmulhost(vector<double> A,vector<double> B, int n, int p);

/*matmul{h}(A,B,n) for h in {ddf,qdf,pdf,odf,daf,hdf} takes nxn matrices A,B of 2,4,5,8,10,16-doubles respectively and computes the product C, all done on the host*/
vector<double> matmulddf(vector<double> A,vector<double> B, int n);
vector<double> matmulqdf(vector<double> A,vector<double> B, int n);
vector<double> matmulodf(vector<double> A,vector<double> B, int n);
vector<double> matmulhdf(vector<double> A,vector<double> B, int n);

/*renormhost is equivalent to renormA but is done purely on the host*/
void renormhost(vector<double> &A,int n,int p);

/*renormhostbig is equivalent to renormbigA but is done purely on the host*/
void renormhostbig(vector<double> &A,int n,int p);

/*squeeze takes a matrix of q-split p doubles and condenses them*/
vector<double> squeeze(vector<double> x,int p,int q);

/*pllsqueeze2 is a parallel variant of squeeze, takes an n dimensional vector x of pq-doubles and places them in an n dimesnional vector of p-doubles y, uses pllsqueezekernel which uses 1 dimensional tiled threading, partitioning n into 64*/
__global__ void pllsqueezekernel(double *x,double *y,int p,int q);
vector<double> pllsqueeze(vector<double> x,int p,int q);

#endif

