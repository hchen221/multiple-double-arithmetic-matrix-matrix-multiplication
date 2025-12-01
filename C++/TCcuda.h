#ifndef _TCFLAT_
#define _TCFLAT_

#include "floatybits.h"

/*
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
*/

/*
Assume matrices have a flat representation by default
mat(n,p) returns a flattened nxn matrix of random p doubles
*/
vector<double> mat(int n, int p, int expmin=0, int expmax=0);

/*
zeros(n,p) returns a matrix of all 0's
*/
vector<double> zeros(int n, int p);

//convmult takes flattened nxn matrices of p-doubles A,B and computes each matrix product A_i*B_j for parts i and j respectively of A and B, then accumulates the result to an nxn matrix of pxp grids C_aux. The matrix products must be tiled with A as 8x8, B and C as 8x4 in order to use tensor core
//Once the matrix products are computed, convadd adds them together as part of the convolution process
__global__ void convmult(double* A,double* B,double* C_aux,int n,int p);
__global__ void convadd(double* C,double* C_aux,int n,int p);
//manualconvmult is a vessel that calls upon the convmult and convadd kernels to compute the product C=A*B where A,B are flattened nxn matrices of p-doubles
vector<double> manualconvmult(vector<double> A,vector<double> B,int n,int p);

//dotconvbutbetter takes in flattened nxn matrices of p-doubles A,B,C and computes C=A*B directly where each inner product involves a convolution on the p-double parts. Executed on nxn blocks of pxp threads
__global__ void dotconvbutbetter(double* A,double* B,double* C,int n,int p);
//directdotconv is a vessel that calls upon the dotconvbutbetter kernel
vector<double> directdotconv(vector<double> A,vector<double> B,int n,int p);

#endif

