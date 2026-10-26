#include "TCcuda.h"
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

#define nd 16
#define bd 4

/*test_diagonal_inv generates an upper triangular matrix U of size nd x nd and tests the kernel for inverting the bd x bd upper triangular blocks along the diagonal then storing them in Unt*/
void test_diagonal_inv() {
    vector<double> U = mat(nd,1,0,0);
    for (int i=1;i<nd;i++) {
        for (int j=0;j<i;j++) {
            U[i*nd+j] = 0;
        }
    }
    cout << "U =";
    for (int i=0;i<nd*nd;i++) {
        if (i%nd==0) {
            cout << endl;
        }
        cout << U[i] << " ";
    }
    cout << endl << endl;
    vector<double> Unt = zeros(nd,nd,1);
    double* U_d;
    cudaMalloc((void**)&U_d,(long long int)nd*(long long int)nd*(long long int)sizeof(double));
    cudaMemcpy(U_d,U.data(),(long long int)nd*(long long int)nd*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    double* Unt_d;
    cudaMalloc((void**)&Unt_d,(long long int)nd*(long long int)nd*(long long int)sizeof(double));
    cudaMemcpy(Unt_d,Unt.data(),(long long int)nd*(long long int)nd*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    int gd = nd/bd;
    dim3 Gdim;
    dim3 Bdim;
    Gdim.x = 1;
    Gdim.y = 1;
    Bdim.x = gd;
    Bdim.y = bd;
    //cout << "Tile " << nd << "x" << nd << " into " << bd << "x" << bd << " size blocks\n\n";
    invert_diagonals<<<Gdim,Bdim>>>(U_d,Unt_d,nd,bd);
    cudaMemcpy(Unt.data(),Unt_d,(long long int)nd*(long long int)nd*(long long int)sizeof(double),cudaMemcpyDeviceToHost);
    cout << "U^{-1} =";
    for (int i=0;i<nd*nd;i++) {
        if (i%nd==0) {
            cout << endl;
        }
        cout << Unt[i] << " ";
    }
    cout << endl << endl;

    vector<double> I_allegedly = zeros(nd,nd,1);
    double* I_device;
    cudaMalloc((void**)&I_device,(long long int)nd*(long long int)nd*(long long int)sizeof(double));
    cudaMemcpy(I_device,I_allegedly.data(),(long long int)nd*(long long int)nd*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    dim3 Gdim_mat;
    dim3 Bdim_mat;
    Gdim_mat.x = gd;
    Gdim_mat.y = gd;
    Bdim_mat.x = bd;
    Bdim_mat.y = bd;
    verify_diagonals<<<Gdim_mat,Bdim_mat>>>(Unt_d,U_d,I_device,nd);
    cudaMemcpy(I_allegedly.data(),I_device,(long long int)nd*(long long int)nd*(long long int)sizeof(double),cudaMemcpyDeviceToHost);
    cout << "I (allegedly) =";
    for (int i=0;i<nd*nd;i++) {
        if (i%nd==0) {
            cout << endl;
        }
        cout << I_allegedly[i] << " ";
    }
    cout << endl << endl;
}

int main() {
    test_diagonal_inv();
    return 0;
}
