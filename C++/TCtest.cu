#include "TCcuda.h"
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
    cout << "A,B in R^{" << n << "x" << n << "}, entries of "<< p << "-doubles\nTiles of size " << nfrag << "\n\n";
    
    vector<double> C1 = manualconvmult(A8,B8,n,4*p,nfrag);
    cout << "Convolutions on matrix products? Computed." << endl;
    vector<double> C2 = directdotconv(A8,B8,n,4*p);
    cout << "Direct dot product convolutions? Calculated." << endl;

    cout << "C[1,1]? (";
    for (int i=0;i<4*p;i++) {
        cout << C1[i];
	if (i<4*p-1) {
	    cout << ",";
	}
    }
    cout << ")" << endl;
    
    double max_err = 0;
    for (int i=0;i<n*n*4*p;i++) {
	if (max_err<abs(C1[i]-C2[i])) {
	    max_err = abs(C1[i]-C2[i]);
	}
    }
    cout << "Max error? " << max_err << endl;

    return 0;

}




