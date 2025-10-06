#include "TCcuda.h"
#include <iostream>

using namespace std;

int main() {
    int p = 4;
    int n = 64;
    int nfrag = 8;
    int expmin = 0;
    int expmax = 0;

    vector<double> A = mat(n,p,expmin,expmax);
    vector<double> B = mat(n,p,expmin,expmax);
    vector<double> A8 = split4pd(A);
    vector<double> B8 = split4pd(B);
    vector<double> C8 = zeros(n,4*p);
    vector<double> C_aux = zeros(n,4*p*4*p);
    cout << "A,B in R^{" << n << "x" << n << "}, entries of "<< p << "-doubles\nTiles of size " << nfrag << "\n\n";
    double* cA;
    double* cB;
    double* cC;
    double* cC_aux;
    double* cC_worse;
    
    cudaMalloc((void**)&cA,n*n*4*p);
    cudaMemcpy(cA,A8.data(),n*n*4*p,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cB,n*n*4*p);
    cudaMemcpy(cB,B8.data(),n*n*4*p,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cC_aux,n*n*4*p*4*p);
    cudaMemcpy(cC_aux,C_aux.data(),n*n*4*p*4*p,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cC,n*n*4*p);
    cudaMemcpy(cC,C8.data(),n*n*4*p,cudaMemcpyHostToDevice);

    dim3 gridsize(4*p,4*p);
    dim3 blocksize(n,n);
    convmult<<<gridsize,blocksize>>>(cA,cB,cC_aux);
    dim3 flatsize(4*p,1);
    convadd<<<flatsize,blocksize>>>(cC_aux,cC);

    vector<double> C8;

    cudaMemcpy(C8.data(),cC,n*n*4*p,cudaMemcpyDeviceToHost);
    
    return 0;
}