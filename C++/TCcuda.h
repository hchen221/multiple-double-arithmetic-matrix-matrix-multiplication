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

/*bigB2(B,n,p) takes an nxn matrix of p-double entries B and returns an np x np matrix where each pxp block corresponds to an entry of B in the following form
  [b_1,b_2,...,b_p]
  [0  ,b_1,...,b_{p-1}]
  [.  ,.  ,.  ..  ]
  [.  ,.  ,.  ,b_1]
 formatter col major 
*/
vector<double> bigB2(vector<double> B,int n,int p);

/*ddf functions copied from PHCpack/src/GPU/Norms/double_double_functions.h*/
double ddf_quick_two_sum ( double a, double b, double *err );
/*
 * DESCRIPTION :
 *   Assuming |a| >= |b|, returns a+b and in err the error.
 *
 * ON ENTRY :
 *   a,b      two doubles: |a| >= |b|.
 *
 * ON RETURN :
 *   s        returned sum of a and b.
 *   err      error value, b - (s - a). */

double ddf_two_sum ( double a, double b, double *err );
/*
 * DESCRIPTION :
 *   Computes fl(a+b) and err(a+b).
 *
 * ON ENTRY :
 *   a,b      two doubles.
 *
 * ON RETURN :
 *   s        approximation for the sum of a and b is returned;
 *   err      error of a + b. */

void ddf_add
 ( double a_hi, double a_lo, double b_hi, double b_lo,
   double *c_hi, double *c_lo );
/*
 * DESCRIPTION : c = a + b.
 *   Adds two double doubles in a (a_hi, a_lo) and b (b_hi, b_lo)
 *   to make the double double c (c_hi, c_lo).
 *
 * ON ENTRY :
 *   a_hi     high part of the double double a;
 *   a_lo     low part of the double double a;
 *   b_hi     high part of the double double b;
 *   b_lo     low part of the double double b.
 *
 * ON RETURN :
 *   c_hi     high part of the double double c;
 *   c_lo     low part of the double double c. */

/*renormbigA takes a matrix of form bigA(A,n,p) and runs on nxn threads in a 1x1 block to renormalize into non overlapping p-doubles component wise*/
__global__ void renormbigA(double* A,int n,int p);

/*renormA takes an nxn matrix A of p-doubles and runs on nxn threads in a 1x1 block to renormalize into non overlapping p-doubles component wise*/
__global__ void renormA(double* A,int n,int p);

/*matmulhost(A,B,n,p) takes nxn matrices A,B of p-doubles and computes the product C, all done on the host*/
vector<double> matmulhost(vector<double> A,vector<double> B, int n, int p);

/*renormhost is equivalent to renormA but is done purely on the host*/
void renormhost(vector<double> &A,int n,int p);

/*renormhostbig is equivalent to renormbigA but is done purely on the host*/
void renormhostbig(vector<double> &A,int n,int p);

#endif

