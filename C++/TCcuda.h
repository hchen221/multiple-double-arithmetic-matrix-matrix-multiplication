#ifndef _TCFLAT_
#define _TCFLAT_

#include "floatybits.h"


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

//dotconvbutbetter takes in flattened nxn matrices of p-doubles A,B,C and computes C=A*B directly where each inner product involves a convolution on the p-double parts. Executed on nxn blocks of pxp threads
__global__ void dotconvbutbetter(double* A,double* B,double* C,int n,int p);
//directdotconv is a vessel that calls upon the dotconvbutbetter kernel
vector<double> directdotconv(vector<double> A,vector<double> B,int n,int p);

#endif

