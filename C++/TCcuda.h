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

__global__ void matmul(double* A,double* B,double* C);

//__global__ void dotconvbutbetter(double* A,double* B,double* C);

//void matconv(double* A,double* B,double* C,int n,int p,int nfrag);

__global__ void convmult(double* A,double* B,double* C_aux);
__global__ void convmult2(double* A,double* B,double* C_aux);
__global__ void convadd(double* C,double* C_aux);

__global__ void dotconvbutbetter(double* A,double* B,double* C);

#endif

