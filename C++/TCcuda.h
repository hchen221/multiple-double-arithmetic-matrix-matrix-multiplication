#ifndef _TCFLAT_
#define _TCFLAT_

#include "floatybits.h"

/*
Assume matrices have a flat representation by default
mat(n,p) returns a flattened nxn matrix of random p doubles
*/
vector<double> mat(int n, int p, int expmin=0, int expmax=0);

/*
zeros(n,p) returns a matrix of all 0's
*/
vector<double> zeros(int n, int p);

/*
C++ doesn't seem to have nice slicing like in Julia, so matslice(A,ind) is a stand-in for A[ind]
*/
vector<double> matslice(vector<double> A,vector<int> ind);

/*
part(x,j,p) returns the jth part of a p-double vector x component wise
*/
vector<double> part(vector<double> x, int j, int p);

/*
Given an nxn matrix A of p doubles flattened
A[ent(n,p,i,j)] returns A[i,j]
A[row(n,p,i)] returns A[i,:]
A[rowslice(n,p,i,j1,j2)] returns A[i,j1:j2]
A[col(n,p,j)] returns A[:,j]
A[frag(n,p,i1,i2,j1,j2)] returns A[i1:i2,j1:j2]
All still in flattened form
The jth part of the p double component wise could be taken simply by x[j:p:end]
*/
vector<int> ent(int n, int p, int i, int j);
vector<int> row(int n, int p, int i);
vector<int> rowslice(int n, int p, int i, int j1, int j2);
vector<int> col(int n, int p, int j);
vector<int> frag(int n, int p, int i1, int i2, int j1, int j2);

//simple kernel for matrix matrix multiplication on flat arrays of size nxn, executed on nlen x nlen tiles of size nfrag x nfrag where nlen*nfrag=n. This is just a stand-in for the Tensor Core kernel used to multiply matrices
__global__ void matmul(double* A,double* B,double* C, int n);

//convmult takes flattened nxn matrices of p-doubles A,B and computes each matrix product A_i*B_j for parts i and j respectively of A and B, then accumulates the result to an nxn matrix of pxp grids C_aux. The matrix products can be done tiled with tile size nfrag
//Once the matrix products are computed, convadd adds them together as part of the convolution process
__global__ void convmult(double* A,double* B,double* C_aux,int n,int p,int nfrag);
__global__ void convadd(double* C,double* C_aux,int n,int p);
//manualconvmult is a vessel that calls upon the convmult and convadd kernels to compute the product C=A*B where A,B are flattened nxn matrices of p-doubles, matrix products computed using tile size nfrag
vector<double> manualconvmult(vector<double> A,vector<double> B,int n,int p, int nfrag);

//dotconvbutbetter takes in flattened nxn matrices of p-doubles A,B,C and computes C=A*B directly where each inner product involves a convolution on the p-double parts. Executed on nxn blocks of pxp threads
__global__ void dotconvbutbetter(double* A,double* B,double* C,int n,int p);
//directdotconv is a vessel that calls upon the dotconvbutbetter kernel
vector<double> directdotconv(vector<double> A,vector<double> B,int n,int p);

#endif

