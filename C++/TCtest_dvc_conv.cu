#include "TCcuda.h"
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

#define nt 4
#define pt 2
#define nfragt 2

/*test_dnc_conv makes an nt x nt matrix B of pt-doubles and tests the generation of a big convolution matrix BB using the device (bigB_dvc in TCcuda.cu) by comparing with the hostt variant (bigB2 in TCcuda.cu) which is accurate*/
void test_dvc_conv() {
    double* B;
    double* BB;
    vector<double> B_src = mat(nt,pt,0,0);
    cudaMalloc((void**)&B,(long long int)nt*(long long int)nt*(long long int)pt*(long long int)sizeof(double));
    cudaMemcpy(B,B_src.data(),(long long int)nt*(long long int)nt*(long long int)pt*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&BB,(long long int)nt*(long long int)nt*(long long int)pt*(long long int)pt*(long long int)sizeof(double));
    for (int i=0;i<pt;i++) {
        cout << B+i << " ";
    }
    cout << endl << endl;

    dim3 gridB;
    dim3 blockB;

    if (nt<64) {
        gridB.x = nt;
        gridB.y = nt;
        blockB.x = 1;
        blockB.y = 1;
    } else {
        gridB.x = 64;
        gridB.y = 64;
        blockB.x = (int)(n/64);
        blockB.y = (int)(n/64);
    }

    bigB_dvc<<<gridB,blockB>>>(B,BB,nt,pt);

    vector<double> BB_out = zeros(nt,nt,pt*pt);
    cudaMemcpy(BB_out.data(),BB,(long long int)nt*(long long int)nt*(long long int)pt*(long long int)pt*(long long int)sizeof(double),cudaMemcpyDeviceToHost);
    for (int i=0;i<nt*nt*pt*pt;i++) {
        cout << BB_out[i] << " ";
        if (i%(nt*pt)==nt*pt-1) {
            cout << endl;
        }
    }
    vector<double> BB_real = bigB2(B_src,nt,pt);
    cout << endl;
    for (int i=0;i<nt*nt*pt*pt;i++) {                                                                                           cout << BB_real[i] << " ";
        if (i%(nt*pt)==nt*pt-1) {                                                                                                   cout << endl;
        }
    }
}

int main() {
    test_dvc_conv();
    return 0;
}
