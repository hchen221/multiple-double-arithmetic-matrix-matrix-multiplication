#include "TCcuda.h"

#include <iostream>
#include <cmath>

using namespace std;

int main() {
    int p = 4;
    int n = 128;
    int nfrag = 16;
    int expmin = 0;
    int expmax = 0;

    vector<double> A = mat(n,p,expmin,expmax);
    vector<double> B = mat(n,p,expmin,expmax);
    vector<double> A8 = split4pd(A);
    vector<double> B8 = split4pd(B);
    vector<vector<double>> A8s = splitp(A8,4*p);
    vector<vector<double>> B8s = splitp(B8,4*p);

    vector<vector<double>> C8s = manualconvmult(A8s,B8s,n,4*p,nfrag);

    cout << "C8? Computed. Method? Matrix convolutions." << endl;
    for (int i=0;i<4*p;i++) {
        cout << C8s[i][0] << ",";
    }
  
    vector<double> C8 = zeros(n,4*p);

    vector<vector<double>> C8worse = directdotconv(A8,B8,C8,n,4*p);
    cout << "\n\nC8? Computed. Method? Direct inner product convolutions." << endl;
    for (int i=0;i<4*p;i++) {
        cout << C8worse[i][0] << ",";
    }

    double max_err = 0;
    for (int i=0;i<4*p;i++) {
        for (int j=0;j<n*n;j++) {
            if (max_err < abs(C8s[i][j]-C8worse[i][j])) {
                max_err = abs(C8s[i][j]-C8worse[i][j]);
            }
        }
    }

    cout << "\n\nMax error? " << max_err << endl;

    return 0;
}
