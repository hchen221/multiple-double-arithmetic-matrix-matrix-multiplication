#ifndef _TCFLAT_
#define _TCFLAT_

#include "floatybits.h"
#include <functional>

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

/*
flatmyconv(x,y,p,f) computes f(x,y) using convolutions. flatconvpart and flatSconv are helper functions
*/
double flatconvpart(vector<double> x, vector<double> y, int p, int i, int j);
double flatSconv(vector<double> x, vector<double> y, int p, int i);
vector<double> flatmyconv(vector<double> x, vector<double> y, int p);

/*
flatTCKernel(A,B,C,n,p) is used to simulate a TensorCore kernel performing A*B where A,B are nxn with p doubles and filling it to C
flatdotapply is a helper function
*/
void flatdotapply(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int i, int j);
void flatTCKernel(vector<double> A, vector<double> B, vector<double> &C, int n, int p);

/*
flatTCfrag(A,B,C,n,p,nfrag,ia,ib,ja,jb) computes the matrix product of the (ia,ja) tile of A and the (ib,jb) tile of B
then adds the result to C[ia:ib,ja:jb]
*/
void flatTCfrag(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int nfrag, int ia, int ib, int ja, int jb);

/*
flatfraginnerproduct(A,B,i,j,nfrag,nlen) interprets A,B as nlen x nlen matrices where each entry is an nfrag x nfrag matrix
It computes the inner product of the ith row of A and jth column of B
*/
void flatfraginnerproduct(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int i, int j, int nfrag, int nlen);

/*
flatmul_fragments(A,B,nfrag,p) performs A*B by partitioning it into tiles of size nfrag, where each entry involves p-double arithmetic
*/
void flatmul_fragments(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int nfrag);

/*
flatmatconv(A,B,C,n,p,f) uses matrix covolutions to compute A*B and fills it to C
*/
void flatmatconvhelp1(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int i, int j, int k);
void flatmatconvhelp2(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int k);
void flatmatconv(vector<double> A, vector<double> B, vector<double> &C, int n, int p);

/*
max_err(A,B) returns the maximum error between A and B
*/
double max_err(vector<double> A,vector<double> B);
#endif
