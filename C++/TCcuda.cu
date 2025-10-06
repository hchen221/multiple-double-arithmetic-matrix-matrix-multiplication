#include "TCcuda.h"
#include "floatybits.h"
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
zeros(n,p) returns an nxn matrix of p-double entries with all 0's
*/
vector<double> zeros(int n, int p) {
    vector<double> A;
    for (int i=0;i<n*n*p;i++) {
        A.push_back(0);
    }
    return A;
}


/*
C++ doesn't seem to have nice slicing like in Julia, so matslice(A,ind) is a stand-in for A[ind]
*/
vector<double> matslice(vector<double> A,vector<int> ind) {
    vector<double> A_sub;
    for (int i=0;i<ind.size();i++) {
        A_sub.push_back(A[ind[i]]);
    }
    return A_sub;
}

/*
part(x,j,p) returns the jth part of a p-double vector x component wise
*/
vector<double> part(vector<double> x, int j, int p) {
    vector<double> xj;
    for (int i=j;i<x.size();i+=p) {
        xj.push_back(x[i]);
    }
    return xj;
}

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
vector<int> ent(int n, int p, int i, int j) {
    vector<int> ind;
    for (int k=i*n*p+j*p;k<=i*n*p+(j+1)*p-1;k++) {
        ind.push_back(k);
    }
    return ind;
}
vector<int> row(int n, int p, int i) {
    vector<int> ind;
    for (int k=i*n*p;k<=(i+1)*n*p-1;k++) {
        ind.push_back(k);
    }
    return ind;
}
vector<int> rowslice(int n, int p, int i, int j1, int j2) {
    vector<int> ind1 = row(n,p,i);
    vector<int> ind;
    for (int k=j1*p;k<=(j2+1)*p-1;k++) {
        ind.push_back(k);
    }
    return ind;
}
vector<int> col(int n, int p, int j) {
    vector<int> ind;
    for (int i=0;i<n;i++) {
        vector<int> ind2 = ent(n,p,i,j);
        ind.insert(ind.end(),ind2.begin(),ind2.end());
    }
    return ind;
}
vector<int> frag(int n, int p, int i1, int i2, int j1, int j2) {
    vector<int> ind;
    for (int i=i1;i<=i2;i++) {
        vector<int> ind2 = rowslice(n,p,i,j1,j2);
        ind.insert(ind.end(),ind2.begin(),ind2.end());
    }
    return ind;
}

/*
matmul!(A,B,C) is a dot product kernel that accumulates the results of A*B to C where A,B,C are nxn matrices of doubles

__kernel__ void matmul(double* A,double* B,double* C) {
    int n = gridDim.x*blockDim.x;
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    for (int k=0;k<n;k++) {
        C[i*n+j] += A[i*n+k]*B[k*n+j];
    }
}
*/
/*
dotconvbutbetter!(A,B,C) runs nxn blocks of px1 threads to do the same as dotconv!(A,B,C), doesn't rely on atomic add

__kernel__ void dotconvbutbetter(double* A,double* B,double* C) {
    int n = gridDim.x;
    int p = blockDim.x;
    int I = blockIdx.x;
    int J = blockIdx.y;
    int i = threadIdx.x;
    for (int k=0;k<n;k++) {
        for (int j=0;j<p;j++) {
            if (j+i-p>0) {
                a = A[I*n*p+k*p+j+i-p];
            } else {
                a = 0;
            }
            b = B[k*n*p+J*p+p-j]
            C[I*n*p+J*p+i] += a*b;
        }
    }
}
*/
/*
void matconv(double* A,double* B,double* C,int n,int p,int nfrag) {
    int nlen = n/nfrag;
    for (int k=0;k<p;k++) {
        double Cij[n*n] = {};
        for (int i=0;i<k;i++) {
            double Ai[n*n] = {};
            double Bj[n*n] = {};
            for (int l=0;l<n*n;l++) {
                Ai[l] = A[i+l*p];
                Bj[l] = B[k-i+l*p];
            }
            dim3 gridsize(nfrag,nfrag);
            dim3 blocksize(nlen,nlen);
            matmul<<<gridsize,blocksize>>>(Ai,Bj,Cij);
        }
        for (int l=0;l<n*n;l++) {
            C[k+l*p] += Cij[l];
        }
    }
}
*/

/*Do p^2 blocks for the multiplication part of the convolution, each block does n^2 threads (maybe reduce it to n threads for a single inner product for testing purposes), have a separate kernel do the adding*/
__kernel__ void convmult(double* A,double* B,double* C_aux) { // A,B are n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts for rows form columns)
    int p = gridDim.x;
    int n = blockDim.x;
    int i = blockIdx.x;
    int j = blockIdx.y;
    int I = threadIdx.x;
    int J = threadIdx.y;
    // Want to compute A_i[I,:]*B_j[:,J], C_aux_{i,j}[I,J]
    for (int K=0;K<n;K++) {
        C_aux[I*n*p*p+J*p*p+i*p+j] += A[I*n*p+K*p+i]*B[K*n*p+J*p+j];
    }
}

__kernel__ void convadd(double* C,double* C_aux) { // C is n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts form rows form columns)
    int p = gridDim.x;
    int n = blockDim.x;
    int i = blockIdx.x;
    int I = threadIdx.x;
    int J = threadIdx.y;
    for (int j=0;j<=i;j++) {
        C[I*n*p+J*p+i] += C_aux[I*n*p*p+J*p*p+j*p+(i-j)];
    }
}