#include <iostream>

extern "C" {
    void dgemm_(
        const char *transA, const char *transb,
        const int *M, const int *N, const int *K,
        const double *alpha, const double *A, const int *lda,
        const double *B, const int *ldb,
        const double *beta, double *C, const int *ldc);

}

int main() {

    constexpr double A[6] = {1,2,3,4,5,6};
    constexpr double B[6] = {1,2,3,4,5,6};
    static double C[4] = {0,0,0,0};

    constexpr char trans = 'N';
    constexpr int M=2, N=2, K=3;
    constexpr double alpha=1.0, beta=0.0;
    dgemm_(&trans, &trans, &M, &N, &K, &alpha, A, &M, B, &K, &beta, C, &M);

    std::cout << "C[0,0]=" << C[0] << " C[0,1]=" << C[1] << "\n";

}
