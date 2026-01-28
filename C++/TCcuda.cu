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
    for (int i=0;i<p;i++) {
	for (int c=0;c<n;c++) {
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


/*ddf functions copied from PHCpack/src/GPU/Norms/double_double_functions.cpp*/
double ddf_quick_two_sum ( double a, double b, double *err )
{
   double s = a + b;
   *err = b - (s - a);
   return s;
}

double ddf_two_sum ( double a, double b, double *err )
{
   double s = a + b;
   double bb = s - a;
   *err = (a - (s - bb)) + (b - bb);
   return s;
}

void ddf_add
 ( double a_hi, double a_lo, double b_hi, double b_lo,
   double *c_hi, double *c_lo )
{
   double s1, s2, t1, t2;

   s1 = ddf_two_sum(a_hi,b_hi,&s2);
   t1 = ddf_two_sum(a_lo,b_lo,&t2);
   s2 += t1;
   s1 = ddf_quick_two_sum(s1,s2,&s2);
   s2 += t2;
   *c_hi = ddf_quick_two_sum(s1,s2,c_lo);
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
    return bigA(C,n,p);
}

void renormhost(vector<double> &A,int n,int p) {
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
