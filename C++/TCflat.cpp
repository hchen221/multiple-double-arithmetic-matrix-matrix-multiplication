#include "TCflat.h"
#include "floatybits.h"

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

/*
flatmyconv(x,y,p,f) computes f(x,y) using convolutions. flatconvpart and flatSconv are helper functions
*/
double flatconvpart(vector<double> x, vector<double> y, int p, int i, int j) {
    double s = 0;
    vector<double> xi = part(x,i,p);
    vector<double> yj = part(y,j,p);
    for (int k=0;k<xi.size();k++) {
        s += xi[k]*yj[k];
    }
    return s;
}
double flatSconv(vector<double> x, vector<double> y, int p, int i) {
    double s = 0;
    for (int j=0;j<i;j++) {
        s += flatconvpart(x,y,p,j,i-j-1);
    }
    return s;
}
vector<double> flatmyconv(vector<double> x, vector<double> y, int p) {
    vector<double> S;
    for (int i=0;i<p;i++) {
        S.push_back(flatSconv(x,y,p,i));
    }
    return S;
}

/*
flatTCKernel(A,B,C,n,p) is used to simulate a TensorCore kernel performing A*B where A,B are nxn with p doubles and filling it to C
flatdotapply is a helper function
*/
void flatdotapply(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int i, int j) {
    vector<int> ind = ent(n,p,i,j);
    vector<double> Cij = flatmyconv(matslice(A,row(n,p,i)),matslice(B,col(n,p,j)),p);
    for (int k=0;k<ind.size();k++) {
        C[ind[k]] += Cij[k];
    }
}
void flatTCKernel(vector<double> A, vector<double> B, vector<double> &C, int n, int p) {
    for (int i=0;i<n;i++) {
        for (int j=0;j<n;j++) {
            flatdotapply(A,B,C,n,p,i,j);
        }
    }
}

/*
flatTCfrag(A,B,C,n,p,nfrag,ia,ib,ja,jb) computes the matrix product of the (ia,ja) tile of A and the (ib,jb) tile of B
then adds the result to C[ia:ib,ja:jb]
*/
void flatTCfrag(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int nfrag, int ia, int ib, int ja, int jb) {
    vector<double> Afrag = matslice(A,frag(n,p,ia*nfrag,(ia+1)*nfrag-1,ja*nfrag,(ja+1)*nfrag-1));
    vector<double> Bfrag = matslice(B,frag(n,p,ib*nfrag,(ib+1)*nfrag-1,jb*nfrag,(jb+1)*nfrag-1));
    vector<int> Cind = frag(n,p,ia*nfrag,(ia+1)*nfrag-1,jb*nfrag,(jb+1)*nfrag-1);
    vector<double> Cfrag;
    for (int k=0;k<Cind.size();k++) {
        Cfrag.push_back(0);
    }
    flatTCKernel(Afrag,Bfrag,Cfrag,nfrag,p);
    for (int k=0;k<Cind.size();k++) {
        C[Cind[k]] += Cfrag[k];
    }
}

/*
flatfraginnerproduct(A,B,i,j,nfrag,nlen) interprets A,B as nlen x nlen matrices where each entry is an nfrag x nfrag matrix
It computes the inner product of the ith row of A and jth column of B
*/
void flatfraginnerproduct(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int i, int j, int nfrag, int nlen) {
    for(int k=0;k<nlen;k++) {
        flatTCfrag(A,B,C,n,p,nfrag,i,k,k,j);
    }
}

/*
flatmul_fragments(A,B,nfrag,p) performs A*B by partitioning it into tiles of size nfrag, where each entry involves p-double arithmetic
*/
void flatmul_fragments(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int nfrag) {
    int nlen = n/nfrag;
    for (int i=0;i<nlen;i++) {
        for (int j=0;j<nlen;j++) {
            flatfraginnerproduct(A,B,C,n,p,i,j,nfrag,nlen);
        }
    }
}

/*
flatmatconv(A,B,C,n,p,f) uses matrix covolutions to compute A*B and fills it to C
*/
void flatmatconvhelp1(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int i, int j, int k) {
    vector<double> Ai = part(A,i,p);
    vector<double> Bj = part(B,j,p);
    vector<double> Ck = zeros(Ai.size(),1);
    flatTCKernel(Ai,Bj,Ck,n,1);
    for (int l=0;l<Ai.size();l++) {
        C[k+l*p] += Ck[l];
    }
}
void flatmatconvhelp2(vector<double> A, vector<double> B, vector<double> &C, int n, int p, int k) {
    for (int i=0;i<k;i++) {
        flatmatconvhelp1(A,B,C,n,p,i,k-1-i,k);
    }
}
void flatmatconv(vector<double> A, vector<double> B, vector<double> &C, int n, int p) {
    for (int k=0;k<p;k++) {
        flatmatconvhelp2(A,B,C,n,p,k);
    }
}

/*
max_err(A,B) returns the maximum error between A and B
*/
double max_err(vector<double> A,vector<double> B) {
    double eps = 0;
    for (int i=0;i<A.size();i++) {
        double eps2 = abs(A[i]-B[i]);
        if (eps2 > eps) {
            eps = eps2;
        }
    }
    return eps;
}
