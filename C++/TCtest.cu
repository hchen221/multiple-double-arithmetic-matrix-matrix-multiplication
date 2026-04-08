#include "TCcuda.h"
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

#define p 2
#define n 256
#define q 4

#define M 8
#define N 8
#define K 4

#define M_GLOBAL n
#define N_GLOBAL n*q*p
#define K_GLOBAL n*q*p

__global__ void matmul(double *a, double *b, double *c) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;

    int cCol = warpN * N;
    int cRow = warpM * M;

    wmma::load_matrix_sync(c_frag,c+cCol+cRow*N_GLOBAL,N_GLOBAL,wmma::mem_row_major);

    for (int k=0;k<K_GLOBAL;k+=K) {
	int aCol = k;
        int aRow = warpM * M;
        int bCol = warpN * N;
        int bRow = k;
	wmma::load_matrix_sync(a_frag, a + aCol + aRow * K_GLOBAL, K_GLOBAL);
        wmma::load_matrix_sync(b_frag, b + bRow + bCol * K_GLOBAL, K_GLOBAL);

	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(c+cCol+cRow*N_GLOBAL,c_frag,N_GLOBAL,wmma::mem_row_major);
}

void test(int expmin,int expmax) {
    vector<double> A = mat(n,p,expmin,expmax);
    vector<double> B = mat(n,p,expmin,expmax);
    vector<double> Aq;
    vector<double> Bq;
    if (q==4) {
        Aq = split4pd(A,p);
        Bq = split4pd(B,p);
    } else if (q==8) {
        Aq = split8pd(A,p);
        Bq = split8pd(B,p);
    }
    //vector<double> A8D = bigA(A8,n,q*p);
    //vector<double> B8D = bigB(B8,n,q*p);
    vector<double> AqD = Aq;
    vector<double> BqD = bigB2(Bq,n,q*p);
    cout << "A,B in R^{" << n << "x" << n << "}, entries of "<< p << "-doubles\n\n";
    
    vector<double> C1q = zeros(n,n,q*p);

    cout << "Setup? Done." << endl;
    
    double d0 = (double)clock();

    double* A_d;
    double* B_d;
    double* C_d;

    cudaMalloc((void**)&A_d,M_GLOBAL*K_GLOBAL*sizeof(double));
    cudaMemcpy(A_d,AqD.data(),M_GLOBAL*K_GLOBAL*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d,K_GLOBAL*N_GLOBAL*sizeof(double));
    cudaMemcpy(B_d,BqD.data(),K_GLOBAL*N_GLOBAL*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&C_d,M_GLOBAL*N_GLOBAL*sizeof(double));
    cudaMemcpy(C_d,C1q.data(),M_GLOBAL*N_GLOBAL*sizeof(double),cudaMemcpyHostToDevice);

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize 128x4 means we have
    // 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (M*blockDim.x/32-1))/(M*blockDim.x/32);
    gridDim.y = (N_GLOBAL + N*blockDim.y-1)/(N*blockDim.y);
    
    float t_raw;
    cudaEvent_t T0,Tf;           // to measure time spent by kernels 
    cudaEventCreate(&T0);
    cudaEventCreate(&Tf);
    cudaEventRecord(T0);
    matmul<<<gridDim,blockDim>>>(A_d,B_d,C_d);
    cudaEventRecord(Tf);
    cudaEventSynchronize(Tf);
    cudaEventElapsedTime(&t_raw,T0,Tf);
    float f_raw = M_GLOBAL*N_GLOBAL*(2*K_GLOBAL-1);

    cudaMemcpy(C1q.data(),C_d,M_GLOBAL*N_GLOBAL*sizeof(double),cudaMemcpyDeviceToHost);
    vector<double> C1 = pllsqueeze(C1q,p,q);
    for (int i=0;i<C1.size();i=i+p) {
        balance(C1,i,i+p-1,53);
    }
    double df = (double)clock();

    cout << "TC? Finished. Raw performance? " << f_raw/t_raw << ". Total time? " << df-d0 << endl;

    float t_CUDA;
    double h0 = (double)clock();
    int nfrag = min(32,n);
    vector<double> C2 = matmulTCnt(A,B,n,nfrag,p,t_CUDA);
    for (int i=0;i<C2.size();i=i+p) {
	balance(C2,i,i+p-1,53);
    }
    double hf = (double)clock();
    float f_CUDA,add_ops,mul_ops;
    if (p==2) {
	mul_ops=23*n*n*n;
	add_ops=20*n*n*(n-1);
    } else if (p==4) { // rest are 23 and 20 as placeholders until table given
	mul_ops=23*n*n*n;
        add_ops=20*n*n*(n-1); 
    } else if (p==8) {
	mul_ops=23*n*n*n;
        add_ops=20*n*n*(n-1);
    } else if (p==16) {
	mul_ops=23*n*n*n;
        add_ops=20*n*n*(n-1);
    }
    f_CUDA = mul_ops+add_ops;

    cout << "CUDA? Finished. Performance? " << f_CUDA/t_CUDA << endl;
    
    cout << "TC C[1,1]? (";
    for (int i=0;i<p;i++) {
        cout << C1[i];
        if (i<p-1) {
            cout << ",";
        }
    }
    cout << ")" << endl;

    cout << "CUDA C[1,1]? (";
    for (int i=0;i<p;i++) {
        cout << C2[i];
	if (i<p-1) {
	    cout << ",";
	}
    }
    cout << ")" << endl;
    
    double max_err = 0;
    for (int i=0;i<n*n*p;i++) {
	if (max_err<abs(C1[i]-C2[i])) {
	    max_err = abs(C1[i]-C2[i]);
	}
    }
    cout << "Max error? " << max_err << endl;
}

int main() {
    int seed = time(NULL);
    srand(seed);
    test(0,0);
    /*
    vector<double> A = mat(2,2,0,0);
    vector<double> A8 = split4pd(A,2);
    int exp0,exp1;
    double fr = frexp(A8[0],&exp0);
    for (int i=1;i<8;i++) {
	double fr = frexp(A8[i],&exp1);
	cout << exp1-exp0 << endl;
	exp0 = exp1;
    }
    */
    
    return 0;
}

