#include "TCflat.h"
#include <iostream>
#include <functional>

using namespace std;

int main() {
    int p = 2;
    int n = 64;
    int nfrag = 8;
    int expmin = 0;
    int expmax = 0;

    vector<double> A = mat(n,p,expmin,expmax);
    vector<double> B = mat(n,p,expmin,expmax);
    vector<double> A8 = split4pd(A);
    vector<double> B8 = split4pd(B);
    cout << "A,B in R^{" << n << "x" << n << "}, entries of "<< p << "-doubles\nTiles of size " << nfrag << "\n\n";
    
    vector<double> C1 = zeros(n,4*p);
    flatTCKernel(A8,B8,C1,n,4*p);
    vector<double> C2 = zeros(n,4*p);
    flatmatconv(A8,B8,C2,n,4*p);

    cout << "C1 applies kernel directly\nC2 uses split matrices\nError of C1,C2 : " << max_err(C1,C2) << endl;
    return 0;
}
