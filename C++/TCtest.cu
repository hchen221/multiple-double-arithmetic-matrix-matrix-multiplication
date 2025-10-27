#include "TCcuda.h"
//#include "TCcuda.cu"
#include <iostream>
#include <cmath>

using namespace std;

void convmultcudant(vector<double> A,vector<double> B,vector<double> &C_aux, int n, int p) {
//cout << "convmult? In. (" << n << "," << p << ")" << endl;	
	for (int i=0;i<p;i++) {
		for (int j=0;j<p;j++) {
			for (int I=0;I<n;I++) {
				for (int J=0;J<n;J++) {					
	for (int K=0;K<n;K++) {
                                                C_aux[I*n*p*p+J*p*p+i*p+j] += A[I*n*p+K*p+i]*B[K*n*p+J*p+j];                                                                             }                                                                }                                                                }                                                                }                                                                }                                                                
//cout << "convmult? done." << endl;
}

void convaddcudant(vector<double> &C,vector<double> &C_aux, int n, int p) {
        for (int i=0;i<p;i++) {
                for (int I=0;I<n;I++) {
                        for (int J=0;J<n;J++) {
                                for (int j=0;j<=i;j++) {
        C[I*n*p+J*p+i] += C_aux[I*n*p*p+J*p*p+j*p+(i-j)];
                                }
                        }
                }
        }
}

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


    //convmult<<<gridsize,blocksize>>>(cA,cB,cC_aux);
    convmult2<<<flatsize,blocksize>>>(cA,cB,cC_aux);
    cudaMemcpy(C_aux.data(),cC_aux,n*n*4*p*4*p*sizeof(double),cudaMemcpyDeviceToHost);
    vector<double> C_aux2 = zeros(n,4*p*4*p);
    convmultcudant(A8,B8,C_aux2,n,4*p);

    double max_err2 = 0;
    for (int k=0;k<n*n*4*p*4*p;k++) {
        //cout << C_aux[k] << ',' << C_aux2[k] << endl;
	    if (abs(C_aux[k]-C_aux2[k]) > max_err2) {
            max_err2 = abs(C_aux[k]-C_aux2[k]);
        }
    }
    cout << "Max err? " << max_err2 << endl;

    return 0;
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




