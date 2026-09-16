#include "TCcuda.h"
#include <iostream>
#include <cmath>
#include <ctime>
using namespace std;

#define p 2
#define n 256
#define q 4
#define pp q*p
#define loop_ct 1 // 1 for correctness, 10000 for performance

#define M 8
#define N 8
#define K 4

#define M_GLOBAL n
#define N_GLOBAL n*pp
#define K_GLOBAL n*pp

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
    if (pp==12) { // mix only works if p=2
        Aq = mixsplit2(A);
        Bq = mixsplit2(B);
    } else if (pp==4*p) {
        Aq = split4pd(A,p);
        Bq = split4pd(B,p);
    } else if (pp==8*p) {
        Aq = split8pd(A,p);
        Bq = split8pd(B,p);
    }
    cout << "A,B in R^{" << n << "x" << n << "}, entries of "<< p << "-doubles\n\n";
    
    vector<double> C1q = zeros(n,n,pp);

    cout << "Setup? Done." << endl;
    
    double d0 = (double)clock();

    double* A_d;
    double* B_d;
    double* BB_d;
    double* C_d;

    cudaMalloc((void**)&A_d,(long long int)M_GLOBAL*(long long int)K_GLOBAL*(long long int)sizeof(double));
    cudaMemcpy(A_d,Aq.data(),(long long int)M_GLOBAL*(long long int)K_GLOBAL*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&B_d,(long long int)M_GLOBAL*(long long int)N_GLOBAL*(long long int)sizeof(double));
    cudaMemcpy(B_d,Bq.data(),(long long int)M_GLOBAL*(long long int)N_GLOBAL*(long long int)sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&BB_d,(long long int)K_GLOBAL*(long long int)N_GLOBAL*(long long int)sizeof(double));
    
    dim3 gridB;
    int nB = n/64;
    gridB.x = nB;
    gridB.y = nB;
    dim3 blockB;
    blockB.x = 64;
    blockB.y = 64;
    bigB_dvc<<<gridB,blockB>>>(B_d,BB_d,n,pp);
    
    //vector<double> BB_h = bigB2(Bq,n,pp);
    //cudaMemcpy(BB_d,BB_h.data(),(long long int)K_GLOBAL*(long long int)N_GLOBAL*(long long int)sizeof(double),cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&C_d,(long long int)M_GLOBAL*(long long int)N_GLOBAL*(long long int)sizeof(double));
    cudaMemcpy(C_d,C1q.data(),(long long int)M_GLOBAL*(long long int)N_GLOBAL*(long long int)sizeof(double),cudaMemcpyHostToDevice);

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
    double t0 = (double)clock();
    for (int i=0;i<loop_ct;i++) {
        matmul<<<gridDim,blockDim>>>(A_d,BB_d,C_d);
    }
    double tf = (double)clock();
    cudaEventRecord(Tf);
    cudaEventSynchronize(Tf);
    cudaEventElapsedTime(&t_raw,T0,Tf);
    long long int f_raw = (long long int)M_GLOBAL*(long long int)N_GLOBAL*(2*(long long int)K_GLOBAL-1);

    cudaMemcpy(C1q.data(),C_d,(long long int)M_GLOBAL*(long long int)N_GLOBAL*(long long int)sizeof(double),cudaMemcpyDeviceToHost);
    cout << "Core? Tensored." << endl;

    float t_squeeze;
    vector<double> C1 = pllsqueeze_old(C1q,p,pp,t_squeeze);
    double df = (double)clock();

    long long int f_squeeze = (long long int)pp*(long long int)((int)(C1q.size()/pp));
    if (p==2) {
	f_squeeze *= 10;
    } else if (p==4) { // 10 is a placeholder
	f_squeeze *= 10;
    } else if (p==8) {
	f_squeeze *= 10;
    } else if (p==16) {
	f_squeeze *= 10;
    }

    cout << "TC? Finished. Raw performance? " << f_raw/t_raw << ". CPU time? " << ((df-d0)/(double)CLOCKS_PER_SEC) << ". For Tensor Core alone? " << ((tf-t0)/(double)CLOCKS_PER_SEC) << endl;

    float t_CUDA;
    double h0 = (double)clock();
    int nfrag = min(32,n);
    vector<double> C2 = matmulTCnt(A,B,n,nfrag,p,loop_ct,t_CUDA);
    double hf = (double)clock();
    float f_CUDA,add_ops,mul_ops;
    if (p==2) {
	mul_ops=23*(long long int)n*(long long int)n*(long long int)n;
	add_ops=20*(long long int)n*(long long int)n*((long long int)n-1);
    } else if (p==4) {
	mul_ops=336*(long long int)n*(long long int)n*(long long int)n;
        add_ops=89*(long long int)n*(long long int)n*((long long int)n-1);
    } else if (p==8) {
	mul_ops=1742*(long long int)n*(long long int)n*(long long int)n;
        add_ops=269*(long long int)n*(long long int)n*((long long int)n-1);
    } else if (p==16) { // 23 and 20 are placeholders
	mul_ops=23*(long long int)n*(long long int)n*(long long int)n;
        add_ops=20*(long long int)n*(long long int)n*((long long int)n-1);
    }
    // values from Table 1 in https://homepages.math.uic.edu/~jan/hilt2020multidoubles.pdf
    f_CUDA = mul_ops+add_ops;

    cout << "CUDA? Finished. Performance? " << f_CUDA/t_CUDA << ". CPU time? " << (hf-h0)/(double)CLOCKS_PER_SEC << endl;
    
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
    double* B;
    double* BB;
    int nt = 2;
    int pt = 4;
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
    */
    return 0;
}
