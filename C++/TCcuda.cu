#include "TCcuda.h"
#include "floatybits.h"
//#include "floatybits.cpp"
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

__global__ void matmul(double* A,double* B,double* C,int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    for (int k=0;k<n;k++) {
	C[i*n+j]+=A[i*n+k]*B[k*n+j];
    }
}

/*convmult executes p^2 threads in a single block where thread (i,j) computes the matrix product of A_i*B_j, where the result gets accumulated to C_aux_{i,j}. n is the size of the matrix, nfrag is the size of the tile for the matrix products, and p is the number of double parts
 */
__global__ void convmult(double* A,double* B,double* C_aux,int n,int p,int nfrag) { // A,B are n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts for rows form columns)
    int i = blockIdx.x;
    int j = blockIdx.y;
    // Want to compute A_i[I,:]*B_j[:,J], C_aux_{i,j}[I,J], capital letters denote matrix entries, lowercase denote parts of the p-double
    if (j <= i) {
        double* Ax = new double[n];
        double* By = new double[n];
        double* Cxy = new double[n];

        for (int I=0;I<n;I++) {
	    for (int J=0;J<n;J++) {
	        Ax[I*n+J] = A[I*n*p+J*p+j]; // take the jth part of A
       	        By[I*n+J] = B[I*n*p+J*p+i-j]; // take the i-j th part of B
	        Cxy[I*n+J] = C_aux[I*n*p*p+J*p*p+i*p+j]; // Locate the i,j position in C_aux
    	    }
        }

        int nlen = n/nfrag;
        dim3 gridSize(nlen,nlen);
        dim3 blockSize(nfrag,nfrag);
        matmul<<<gridSize,blockSize>>>(Ax,By,Cxy,n);

        for (int I=0;I<n;I++) {
            for (int J=0;J<n;J++) {
                C_aux[I*n*p*p+J*p*p+i*p+j] = Cxy[I*n+J]; // Locate the i,j position in C_aux
            }
        }
    }
    __syncthreads();

}

__global__ void convadd(double* C,double* C_aux) { // C is n^2*p (parts form rows form columns), C_aux is n^2*p^2 (row parts form column parts form rows form columns)
    int p = gridDim.x;
    int n = blockDim.x;
    int i = blockIdx.x;
    int I = threadIdx.x;
    int J = threadIdx.y;
    for (int j=0;j<p;j++) {
	if (j<=i) {
	    C[I*n*p+J*p+i] += C_aux[I*n*p*p+J*p*p+j*p+(i-j)];
	} else {
            C[I*n*p+J*p+i] += 0;
	}
	__syncthreads();
    }

}

/*This function treats the matrices of p-doubles as a vector of the p matrices in order to perform the convolution
 */
/*
vector<vector<double>> manualconvmult(vector<vector<double>> A,vector<vector<double>> B,int n,int p, int nfrag) {
    vector<vector<double>> C;
    int nlen = n/nfrag;
    for (int i=0;i<p;i++) {
	vector<double> Ci = zeros(n,1);
	C.push_back(Ci);
	for (int j=0;j<=i;j++) {
	    double* Ax;
	    double* By;
	    double* Cxy;
            
	    cudaMalloc((void**)&Ax,n*n*sizeof(double));
	    cudaMemcpy(Ax,A[j].data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
	    cudaMalloc((void**)&By,n*n*sizeof(double));
	    cudaMemcpy(By,B[i-j].data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
	    cudaMalloc((void**)&Cxy,n*n*sizeof(double));
	    cudaMemcpy(Cxy,C[i].data(),n*n*sizeof(double),cudaMemcpyHostToDevice);

	    dim3 gridSize(nlen,nlen);
	    dim3 blockSize(nfrag,nfrag);
	    matmul<<<gridSize,blockSize>>>(Ax,By,Cxy,n);

	    cudaMemcpy(C[i].data(),Cxy,n*n*sizeof(double),cudaMemcpyDeviceToHost);
	}
    }
    return C;
}
*/

/*This is a naive implementation which computes the convolutions within the kernel, using nxn blocks and px1 threads per block. Use for comparison purposes*/
__global__ void dotconvbutbetter(double* A,double* B,double* C) {
    int n = gridDim.x;
    int p = blockDim.x;
    int I = blockIdx.x;
    int J = blockIdx.y;
    int i = threadIdx.x;
    for (int k=0;k<n;k++) {
        for (int j=0;j<p;j++) {
            double a,b;
            if (j<=i) {
                a = A[I*n*p+k*p+j];
                b = B[k*n*p+J*p+(i-j)];
            } else {
                a = 0;
                b = 0;
            }
            C[I*n*p+J*p+i] += a*b;
        }
    }
}

vector<vector<double>> directdotconv(vector<double> A,vector<double> B,vector<double> C, int n, int p) {
    double* Ad;
    double* Bd;
    double* Cd;

    cudaMalloc((void**)&Ad,n*n*p*sizeof(double));
    cudaMalloc((void**)&Bd,n*n*p*sizeof(double));
    cudaMalloc((void**)&Cd,n*n*p*sizeof(double));
    cudaMemcpy(Ad,A.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,B.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Cd,C.data(),n*n*p*sizeof(double),cudaMemcpyHostToDevice);

    dim3 gridSize(n,n);
    dim3 blockSize(p,1);
    dotconvbutbetter<<<gridSize,blockSize>>>(Ad,Bd,Cd);

    cudaMemcpy(C.data(),Cd,n*n*p*sizeof(double),cudaMemcpyDeviceToHost);

    return splitp(C,p);
}

