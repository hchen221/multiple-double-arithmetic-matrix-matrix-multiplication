#include "TCcuda.h"
#include "TCcuda.cu"
#include <iostream>
#include <cmath>

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
    vector<double> C_worse = zeros(n,4*p);
    cout << "A,B in R^{" << n << "x" << n << "}, entries of "<< p << "-doubles\nTiles of size " << nfrag << "\n\n";
    double* cA;
    double* cB;
    double* cC;
    double* cC_aux;
    double* cC_worse;

    cudaMalloc((void**)&cA,n*n*4*p*sizeof(double));
    cudaMemcpy(cA,A8.data(),n*n*4*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cB,n*n*4*p*sizeof(double));
    cudaMemcpy(cB,B8.data(),n*n*4*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cC_aux,n*n*4*p*4*p*sizeof(double));
    cudaMemcpy(cC_aux,C_aux.data(),n*n*4*p*4*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cC,n*n*4*p*sizeof(double));
    cudaMemcpy(cC,C8.data(),n*n*4*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cC_worse,n*n*4*p*sizeof(double));
    cudaMemcpy(cC_worse,C_worse.data(),n*n*4*p*sizeof(double),cudaMemcpyHostToDevice);

    dim3 gridsize(4*p,4*p);
    dim3 blocksize(n,n);
    dim3 flatsize(4*p,1);


    convmult<<<gridsize,blocksize>>>(cA,cB,cC_aux);
    convadd<<<flatsize,blocksize>>>(cC_aux,cC);

    dotconvbutbetter<<<blocksize,flatsize>>>(cA,cB,cC_worse);

    cudaMemcpy(C8.data(),cC,n*n*4*p*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(C_worse.data(),cC_worse,n*n*4*p*sizeof(double),cudaMemcpyDeviceToHost);

    cout << "C8? Complete. C_worse? Complete." << endl;

    double max_err = 0;
    for (int k=0;k<n*n*4*p;k++) {
        if (abs(C8[k]-C_worse[k]) > max_err) {
            max_err = abs(C8[k]-C_worse[k]);
        }
    }
    cout << "Max err? " << max_err << endl;

    return 0;

}



