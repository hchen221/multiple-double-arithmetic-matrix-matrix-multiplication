#include <iostream>
#include <vector>
#include <vector_types.h>

using namespace std;

__global__ void compmul(double* A,double* B,double* C) {
    C[blockIdx.x*blockDim.x+threadIdx.x] = A[blockIdx.x*blockDim.x+threadIdx.x]*B[blockIdx.x*blockDim.x+threadIdx.x];
}

int main() {
    int n = 6;
    int p = 2;
    vector<double> A;
    vector<double> B;
    vector<double> C;

    for (int i=0;i<n*p;i++) {
        A.push_back(i+1);
        B.push_back(n*p-i);
        C.push_back(0);
    }

    //double* A_aux = A.data();
    //double* B_aux = B.data();
    //double* C_aux = C.data();

    cout << "Vectors? Defined. Entries? Filled." << endl;
    double* cA;
    double* cB;
    double* cC;

    cout << "Arrays? Stated." << endl;

    cudaMalloc((void**)&cA,n*p*sizeof(double));
    cudaMemcpy(cA,A.data(),n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cB,n*p*sizeof(double));
    cudaMemcpy(cB,B.data(),n*p*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cC,n*p*sizeof(double));
    cudaMemcpy(cC,C.data(),n*p*sizeof(double),cudaMemcpyHostToDevice);

    cout << "Memory? Allocated. Data? Copied." << endl;

    dim3 grid(p,1,1);
    dim3 blox(n,1,1);
    compmul<<<grid,blox>>>(cA,cB,cC);

    cudaMemcpy(C.data(),cC,n*p*sizeof(double),cudaMemcpyDeviceToHost);

    for (int i=0;i<n*p;i++) {
        cout << A[i] << "*" << B[i] << "=" << C[i] << endl;
    }

    return 0;
}
