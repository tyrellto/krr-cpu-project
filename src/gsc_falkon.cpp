#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>
#include <utility>
#include <omp.h>
#include <algorithm>

// #include <benchmark/benchmark.h>
using namespace std;

struct Vector {
    std::vector<double> data;
    int size;

    Vector(int n) : data(n, 0.0), size(n) {}
    Vector(const std::vector<double>& v) : data(v), size((int)v.size()) {}

    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }
};

struct Matrix {
    int rows, cols;
    std::vector<double> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r*c, 0.0) {}

    double& operator()(int r, int c) { return data[r * cols + c];}
    const double& operator()(int r, int c) const { return data[r * cols + c];}
};

Vector matVec(const Matrix &M, const Vector &v) {
    assert(M.cols == v.size);
    Vector out(M.rows);
    for(int i = 0; i < M.rows; i++){
        double sum = 0.0;
        for(int j = 0; j < M.cols; j++){
            sum += M(i, j) * v[j];
        }
        out[i] = sum;
    }
    return out;
}
// Vector matVec(const Matrix &M, const Vector &v) {
//     assert(M.cols == v.size);
//     Vector out(M.rows);
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < M.rows; i++){
//         double sum = 0.0;
//         for (int j = 0; j < M.cols; j++){
//             sum += M(i, j) * v[j];
//         }
//         out[i] = sum;
//     }
//     return out;
// }

// double dot(const Vector &a, const Vector &b) {
//     assert(a.size == b.size);
//     double sum = 0.0;
//     for(int i = 0; i < a.size; i++){
//         sum += a[i] * b[i];
//     }
//     return sum;
// }
double dot(const Vector &a, const Vector &b) {
    assert(a.size == b.size);
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < a.size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Vector operator*(double scalar, const Vector &v) {
//     Vector out(v.size);
//     for(int i = 0; i < v.size; i++){
//         out[i] = scalar * v[i];
//     }
//     return out;
// }

// Vector operator+(const Vector &a, const Vector &b) {
//     assert(a.size == b.size);
//     Vector out(a.size);
//     for(int i = 0; i < a.size; i++){
//         out[i] = a[i] + b[i];
//     }
//     return out;
// }

// Vector operator-(const Vector &a, const Vector &b) {
//     assert(a.size == b.size);
//     Vector out(a.size);
//     for(int i = 0; i < a.size; i++){
//         out[i] = a[i] - b[i];
//     }
//     return out;
// }
Vector operator*(double scalar, const Vector &v) {
    Vector out(v.size);
    // #pragma omp parallel for schedule(static)
    #pragma omp simd
    for (int i = 0; i < v.size; i++){
        out[i] = scalar * v[i];
    }
    return out;
}

Vector operator+(const Vector &a, const Vector &b) {
    assert(a.size == b.size);
    Vector out(a.size);
    // #pragma omp parallel for schedule(static)
    #pragma omp simd
    for (int i = 0; i < a.size; i++){
        out[i] = a[i] + b[i];
    }
    return out;
}

Vector operator-(const Vector &a, const Vector &b) {
    assert(a.size == b.size);
    Vector out(a.size);
    // #pragma omp parallel for schedule(static)
    #pragma omp simd
    for (int i = 0; i < a.size; i++){
        out[i] = a[i] - b[i];
    }
    return out;
}

// RBF Gaussian kernel
// double rbf_kernel(const std::vector<double> &x,
//                   const std::vector<double> &z,
//                   double gamma) {
//     double sumSq = 0.0;
//     assert(x.size() == z.size());
//     for(int i = 0; i < (int)x.size(); i++){
//         double diff = x[i] - z[i];
//         sumSq += diff * diff;
//     }
//     return std::exp(-gamma * sumSq);
// }

// Matrix computeKernelMatrix(const std::vector<std::vector<double>> &X1,
//                            const std::vector<std::vector<double>> &X2,
//                            double gamma) {
//     Matrix K(X1.size(), X2.size());
//     for(int i = 0; i < (int)X1.size(); i++){
//         for(int j = 0; j < (int)X2.size(); j++){
//             K(i, j) = rbf_kernel(X1[i], X2[j], gamma);
//         }
//     }
//     return K;
// }

double rbf_kernel(const std::vector<double> &x,
                  const std::vector<double> &z,
                  double gamma) {
    double sumSq = 0.0;
    assert(x.size() == z.size());
    // Use OpenMP SIMD reduction to vectorize the summation.
    #pragma omp simd reduction(+:sumSq)
    for (size_t i = 0; i < x.size(); i++) {
        double diff = x[i] - z[i];
        sumSq += diff * diff;
    }
    return std::exp(-gamma * sumSq);
}
Matrix computeKernelMatrix(const std::vector<std::vector<double>> &X1,
                           const std::vector<std::vector<double>> &X2,
                           double gamma) {
    // Make sure there is at least one sample and one feature.
    assert(!X1.empty() && !X2.empty());
    size_t featureSize = X1[0].size();
    Matrix K(X1.size(), X2.size());
    // Parallelize over both loops.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < (int)X1.size(); i++){
        for (int j = 0; j < (int)X2.size(); j++){
            // Here the assertion in rbf_kernel will trigger if dimensions don't match.
            K(i, j) = rbf_kernel(X1[i], X2[j], gamma);
        }
    }
    return K;
}
Matrix computeKernelMatrixBlocked(const Matrix &X1,
                                  const Matrix &X2,
                                  double gamma,
                                  int blockSize) 
{
    int n1 = X1.rows;  // number of samples in X1
    int n2 = X2.rows;  // number of samples in X2
    int d = X1.cols;   // feature dimension (assumed to be same for X1 and X2)
    
    Matrix K(n1, n2);

    // Block over rows (i) and columns (j) of the kernel matrix.
    #pragma omp parallel for schedule(dynamic)
    for (int i0 = 0; i0 < n1; i0 += blockSize) {
        for (int j0 = 0; j0 < n2; j0 += blockSize) {
            // Process a block from i0 to min(n1, i0+B) and j0 to min(n2, j0+B)
            for (int i = i0; i < std::min(n1, i0 + blockSize); i++) {
                for (int j = j0; j < std::min(n2, j0 + blockSize); j++) {
                    // Compute rbf_kernel between sample i from X1 and sample j from X2.
                    double sumSq = 0.0;
                    for (int k = 0; k < d; k++) {
                        double diff = X1(i, k) - X2(j, k);
                        sumSq += diff * diff;
                    }
                    K(i, j) = std::exp(-gamma * sumSq);
                }
            }
        }
    }
    return K;
}

bool cholesky_inplace(Matrix &M) {
    assert(M.rows == M.cols);
    int n = M.rows;
    for (int k = 0; k < n; k++) {
        if (M(k, k) <= 0.0) return false; // not positive definite

        // Compute the square root of the diagonal element.
        M(k, k) = std::sqrt(M(k, k));

        // Update the k-th column (below the diagonal).
        // #pragma omp parallel for schedule(static)
        #pragma omp simd
        for (int i = k + 1; i < n; i++) {
            M(i, k) /= M(k, k);
        }

        // Update the remaining submatrix.
        // Each column j (from k+1 to n-1) can be updated independently.
        #pragma omp parallel for schedule(static)
        for (int j = k + 1; j < n; j++) {
            for (int i = j; i < n; i++) {
                M(i, j) -= M(i, k) * M(j, k);
            }
        }
    }

    // Set the upper-triangular part to zero.
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n; r++) {
        for (int c = r + 1; c < n; c++) {
            M(r, c) = 0.0;
        }
    }
    return true;
}

// solve L * x = b, given a lower-triangular L
Vector forward_sub(const Matrix &L, const Vector &b) {
    int n = L.rows;
    Vector x(n);
    for(int i = 0; i < n; i++){
        double sum = b[i];
        for (int k = 0; k < i; k++){
            sum -= L(i, k) * x[k];
        }
        x[i] = sum / L(i, i);
    }
    return x;
}

// solve L^T * x = b, given a lower-triangular L
Vector backward_sub(const Matrix &L, const Vector &b) {
    int n = L.rows;
    Vector x(n);
    for(int i = n-1; i >= 0; i--){
        double sum = b[i];
        for (int k = i+1; k < n; k++){
            sum -= L(k, i) * x[k]; // L^T(i,k) = L(k,i)
        }
        x[i] = sum / L(i, i);
    }
    return x;
}
// // solve L * x = b, given a lower-triangular L
// Vector forward_sub(const Matrix &L, const Vector &b) {
//     int n = L.rows;
//     Vector x(n);
//     for (int i = 0; i < n; i++) {
//         double sum = b[i];
//         // The inner loop can be vectorized since there is a reduction over k.
//         #pragma omp simd reduction(-:sum)
//         for (int k = 0; k < i; k++) {
//             sum -= L(i, k) * x[k];
//         }
//         x[i] = sum / L(i, i);
//     }
//     return x;
// }

// // solve L^T * x = b, given a lower-triangular L
// Vector backward_sub(const Matrix &L, const Vector &b) {
//     int n = L.rows;
//     Vector x(n);
//     for (int i = n - 1; i >= 0; i--) {
//         double sum = b[i];
//         // Here too, vectorize the inner loop over k with a reduction.
//         #pragma omp simd reduction(-:sum)
//         for (int k = i + 1; k < n; k++) {
//             sum -= L(k, i) * x[k]; // Note: L^T(i,k) = L(k,i)
//         }
//         x[i] = sum / L(i, i);
//     }
//     return x;
// }

typedef std::function<Vector(const Vector &)> LinOp;

Vector conjugateGradient(const LinOp &Aop,
                         const Vector &b,
                         int maxIters,
                         double tol=1e-7,
                         const Vector *x0=nullptr)
{
    int n = b.size;
    Vector x(n), r(n), p(n);

    if(x0) x = *x0;
    else x = Vector(n);

    r = b - Aop(x);
    p = r;
    double rr_old = dot(r, r);

    for(int i = 0; i < maxIters; i++){
        Vector Ap = Aop(p);
        double alpha = rr_old / dot(p, Ap);

        for(int j = 0; j < n; j++){
            x[j] += alpha * p[j];
        }

        for(int j = 0; j < n; j++){
            r[j] -= alpha * Ap[j];
        }

        double rr_new = dot(r, r);
        if(std::sqrt(rr_new) < tol) break;

        double beta = rr_new / rr_old;

        for(int j = 0; j < n; j++){
            p[j] = r[j] + beta * p[j];
        }
        rr_old = rr_new;
    }
    return x;
}
// Vector conjugateGradient(const LinOp &Aop,
//                          const Vector &b,
//                          int maxIters,
//                          double tol=1e-7,
//                          const Vector *x0=nullptr)
// {
//     int n = b.size;
//     Vector x(n), r(n), p(n);

//     if(x0)
//         x = *x0;
//     else 
//         x = Vector(n);

//     r = b - Aop(x);
//     p = r;
//     double rr_old = dot(r, r);

//     for(int i = 0; i < maxIters; i++){
//         Vector Ap = Aop(p);
//         double alpha = rr_old / dot(p, Ap);

//         // Update x: x[j] += alpha * p[j]
//         #pragma omp parallel for schedule(static)
//         for(int j = 0; j < n; j++){
//             x[j] += alpha * p[j];
//         }

//         // Update r: r[j] -= alpha * Ap[j]
//         #pragma omp parallel for schedule(static)
//         for(int j = 0; j < n; j++){
//             r[j] -= alpha * Ap[j];
//         }

//         double rr_new = dot(r, r);
//         if(std::sqrt(rr_new) < tol)
//             break;

//         double beta = rr_new / rr_old;

//         // Update p: p[j] = r[j] + beta * p[j]
//         #pragma omp parallel for schedule(static)
//         for(int j = 0; j < n; j++){
//             p[j] = r[j] + beta * p[j];
//         }

//         rr_old = rr_new;
//     }
//     return x;
// }

struct Preconditioner {
    Matrix T; // cholesky factor of Kmm
    Matrix A; // cholesky factor (1/m)*T*D*T^T + lambda I
};

// WeightedPreconditioner(Xm, ym, alpha, lambda)
// Kmm = k(Xm, Xm)
// z = Kmm * alpha (predictions at the subset points)
// T = chol(kmm)
// D = diag( second derivatives of loss(z[i], ym[i]))
// Kmm <- (1/m)*T*D*T^T + lambda I
// A = chol(Kmm)
// Preconditioner weightedPreconditioner(const Matrix &Kmm,
//                                       const Vector &ym,
//                                       const Vector &alpha,
//                                       double lambda,
//                                       std::function<double(double, double)> secondDerivLoss)
// {
//     int m = Kmm.rows;
//     // 1) z = Kmm * alpha
//     Vector z(m);
//     for(int i = 0; i < m; i++){
//         double sum = 0.0;
//         for(int j = 0; j < m; j++){
//             sum += Kmm(i, j) * alpha[j];
//         }
//         z[i] = sum;
//     }
//     // 2) T = chol(kmm) (copy Kmm first)
//     Matrix Tmat = Kmm;
//     bool ok = cholesky_inplace(Tmat);
//     if(!ok) std::cerr << "Cholesky failed in WeightedPreconditioner.\n";

//     // 3) Build D from second derivatives
//     Vector diagD(m);
//     for(int i = 0; i < m; i++){
//         diagD[i] = secondDerivLoss(z[i], ym[i]);
//     }

//     // 4) Form (1/m)*T * D * T^T
//     Matrix M = Matrix(m, m);
//     // M = (1/m)*(T*D*T^T)
//     // T is lower-tri => for each col, we scale by D, then do T^T?
//     // We will do a naive triple loop for clarity:
//     // Let L = T, we want L * D * L^T => M
//     for(int i = 0; i < m; i++){
//         for(int j = 0; j < m; j++){
//             double sum = 0.0;
//             for(int k = 0; k < m; k++){
//                 // D is diagonal => D(k,k) = diagD[k]
//                 sum += Tmat(i, k) * diagD[k] * Tmat(j, k);
//             }
//             M(i, j) = (1.0/m) * sum;
//         }
//     }

//     for(int i = 0; i < m; i++){
//         M(i, i) += lambda;
//     }

//     Matrix Amat = M;
//     ok = cholesky_inplace(Amat);
//     if(!ok) std::cerr << "Cholesky failed in WeightedPreconditioner A.\n";

//     Preconditioner prec{Tmat, Amat};
//     return prec;
// }
Preconditioner weightedPreconditioner(const Matrix &Kmm,
                                      const Vector &ym,
                                      const Vector &alpha,
                                      double lambda,
                                      std::function<double(double, double)> secondDerivLoss)
{
    int m = Kmm.rows;
    // 1) z = Kmm * alpha
    Vector z(m);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++){
        double sum = 0.0;
        for (int j = 0; j < m; j++){
            sum += Kmm(i, j) * alpha[j];
        }
        z[i] = sum;
    }

    // 2) T = chol(Kmm) (copy Kmm first)
    Matrix Tmat = Kmm;
    bool ok = cholesky_inplace(Tmat);
    if (!ok) std::cerr << "Cholesky failed in WeightedPreconditioner.\n";

    // 3) Build D from second derivatives
    Vector diagD(m);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++){
        diagD[i] = secondDerivLoss(z[i], ym[i]);
    }

    // 4) Form (1/m)*T * D * T^T
    Matrix M(m, m);
    // Parallelize over both i and j using collapse
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            double sum = 0.0;
            for (int k = 0; k < m; k++){
                // D is diagonal: diagD[k] = D(k,k)
                sum += Tmat(i, k) * diagD[k] * Tmat(j, k);
            }
            M(i, j) = (1.0 / m) * sum;
        }
    }

    // 5) Add lambda to the diagonal of M.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++){
        M(i, i) += lambda;
    }

    // 6) Factorize M using Cholesky.
    Matrix Amat = M;
    ok = cholesky_inplace(Amat);
    if (!ok) std::cerr << "Cholesky failed in WeightedPreconditioner A.\n";

    Preconditioner prec{Tmat, Amat};
    return prec;
}

// Weightedfalkon
// X, y: full dataset
// Xm, ym, subset
// K(X, Xm), K(Xm, X) precomputed or compute on the fly
// 'alpha0' can be an initial guess for CG
// 't' = CG iterations
// secondDerivLoss: function to get diag(D)
// Vector weightedFalkon(const Matrix &Kxm,
//                       const Matrix &Kmm,
//                       const Vector &y,
//                       const Vector &ym,
//                       double lambda,
//                       int n,
//                       int t,
//                       const Vector &alpha0,
//                       std::function<double(double, double)> secondDerivLoss)
// {
//     Preconditioner prec = weightedPreconditioner(Kmm, ym, alpha0, lambda, secondDerivLoss);
//     const Matrix &T = prec.T;
//     const Matrix &A = prec.A;
//     int m = Kmm.rows;

//     // 2) Define LinOp(beta)
//     // LinOp(beta) = A^(-T)* (K(Xm,X)* D * K(X,Xm)* T^(-1)* A^(-1)* beta) + lambda n * (A^(-1)*beta)

//     auto solveA = [&](const Vector &b) {
//         // Solve A * x = b => x = forward_sub(A, b), then backward_sub
//         Vector x = forward_sub(A, b);
//         x = backward_sub(A, x);
//         return x; 
//     };

//     auto solveAT = [&](const Vector &b) {
//         // Solve A^T * x = b => use backward_sub then forward_sub w.r.t. A
//         // Because A is lower-triangular
//         // => x = forward_sub(A^T, b) but A^T is upper => ...
//         // simpler to do x = backward_sub(A, b) then forward_sub(A, x)
//         Vector x = backward_sub(A, b);
//         x = forward_sub(A, x);
//         return x;
//     };

//     auto solveT = [&](const Vector &b) {
//         Vector x = forward_sub(T, b);
//         x = backward_sub(T, x);
//         return x;
//     };

//     auto solveTT = [&](const Vector &b) {
//         Vector x = backward_sub(T, b);
//         x = forward_sub(T, x);
//         return x;
//     };

//     // Precompute Kxm^T for convenience => Kmx
//     Matrix Kmx(Kxm.cols, Kxm.rows);
//     for(int i = 0; i < Kxm.rows; i++){
//         for(int j = 0; j < Kxm.cols; j++){
//             Kmx(j, i) = Kxm(i, j);
//         }
//     }

//     auto LinOp = [&](const Vector &beta) -> Vector {
//         // Step 1) v = A^(-1)* beta
//         Vector v = solveA(beta);

//         // Step 2) z = Kxm * beta => predictions on full dataset (size n)
//         // Kxm is n x m, beta is m
//         Vector z(n);
//         for(int i = 0; i < n; i++){
//             double sum = 0.0;
//             for(int j = 0; j < m; j++){
//                 sum += Kxm(i, j) * beta[j];
//             }
//             z[i] = sum;
//         }

//         // Step 3) Build D = diag(secondDerivLoss(z[i],y[i])), i=1..n
//         // Then mulitply Kmx * d * z or so.
//         // c = Kmx * (D * z)
//         // => let w = D*z => w[i] = D[i]*z[i]
//         Vector w(n);
//         for(int i = 0; i < n; i++){
//             double sd = secondDerivLoss(z[i], y[i]);
//             w[i] = sd * z[i];
//         }

//         // c = Kmx * w (Kmx is mx n, w is n => c is m)
//         Vector c(m);
//         for(int i = 0; i < m; i++){
//             double sum = 0.0;
//             for(int j = 0; j < n; j++){
//                 sum += Kmx(i, j) * w[j];
//             }
//             c[i] = sum;
//         }

//         // c <- T^(-1) * c
//         c = solveT(c);

//         // c <- A^(-T) * c
//         c = solveAT(c);

//         // plus lambda n v => total is c + lambda n v
//         // we need to do c = c + lambda * n * v
//         for(int i = 0; i < m; i++){
//             c[i] += (lambda * n) * v[i];
//         }
//         return c;
//     };

//     // 3) Build RHS: R = A^(-T) * Kmx * y
//     // c = Kmx * y => c in R^m
//     Vector c(m);
//     for(int i = 0; i < m; i++){
//         double sum = 0.0;
//         for(int j = 0; j < n; j++){
//             sum += Kmx(i, j) * y[j];
//         }
//         c[i] = sum;
//     }

//     c = solveT(c);

//     Vector R = solveAT(c);

//     // 4) Solve LinOp(beta) = R using CG
//     Vector beta_init(m);
//     for(int i = 0; i < m; i++){
//         beta_init[i] = alpha0[i];
//     }

//     Vector beta = conjugateGradient(LinOp, R, t, 1e-7, &beta_init);

//     // 5) final solution
//     beta = solveA(beta);
//     beta = solveT(beta);

//     return beta;
// }


Vector weightedFalkon(const Matrix &Kxm,
                      const Matrix &Kmm,
                      const Vector &y,
                      const Vector &ym,
                      double lambda,
                      int n,
                      int t,
                      const Vector &alpha0,
                      std::function<double(double, double)> secondDerivLoss)
{
    // Get preconditioner
    Preconditioner prec = weightedPreconditioner(Kmm, ym, alpha0, lambda, secondDerivLoss);
    const Matrix &T = prec.T;
    const Matrix &A = prec.A;
    int m = Kmm.rows;

    // 1) Precompute Kxm^T, i.e. Kmx of dimensions (m x n)
    Matrix Kmx(Kxm.cols, Kxm.rows);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < Kxm.rows; i++){
        for (int j = 0; j < Kxm.cols; j++){
            Kmx(j, i) = Kxm(i, j);
        }
    }

    // Define lambda functions for solving with A and T
    auto solveA = [&](const Vector &b) {
        Vector x = forward_sub(A, b);
        x = backward_sub(A, x);
        return x; 
    };

    auto solveAT = [&](const Vector &b) {
        Vector x = backward_sub(A, b);
        x = forward_sub(A, x);
        return x;
    };

    auto solveT = [&](const Vector &b) {
        Vector x = forward_sub(T, b);
        x = backward_sub(T, x);
        return x;
    };

    auto solveTT = [&](const Vector &b) {
        Vector x = backward_sub(T, b);
        x = forward_sub(T, x);
        return x;
    };

    // Define the LinOp (a lambda mapping Vector->Vector)
    auto LinOp = [&](const Vector &beta) -> Vector {
        // Step 1) v = A^(-1)* beta
        Vector v = solveA(beta);

        // Step 2) z = Kxm * beta  (Kxm is n x m, beta is m-vector)
        Vector z(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++){
            double sum = 0.0;
            for (int j = 0; j < m; j++){
                sum += Kxm(i, j) * beta[j];
            }
            z[i] = sum;
        }

        // Step 3) Build D and compute w = D * z, where w[i] = secondDerivLoss(z[i], y[i]) * z[i]
        Vector w(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++){
            double sd = secondDerivLoss(z[i], y[i]);
            w[i] = sd * z[i];
        }

        // Step 4) c = Kmx * w   (Kmx is m x n, w is n-vector)
        Vector c(m);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++){
            double sum = 0.0;
            for (int j = 0; j < n; j++){
                sum += Kmx(i, j) * w[j];
            }
            c[i] = sum;
        }

        // Step 5) c <- T^(-1) * c
        c = solveT(c);
        // Step 6) c <- A^(-T) * c
        c = solveAT(c);

        // Step 7) c = c + lambda*n*v  (elementwise update)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++){
            c[i] += (lambda * n) * v[i];
        }
        return c;
    };

    // 3) Build RHS: R = A^(-T) * (Kmx * y) computed in two steps.
    Vector c_rhs(m);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++){
        double sum = 0.0;
        for (int j = 0; j < n; j++){
            sum += Kmx(i, j) * y[j];
        }
        c_rhs[i] = sum;
    }
    c_rhs = solveT(c_rhs);
    Vector R = solveAT(c_rhs);

    // 4) Solve LinOp(beta) = R using conjugate gradient (CG)
    Vector beta_init(m);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++){
        beta_init[i] = alpha0[i];
    }
    Vector beta = conjugateGradient(LinOp, R, t, 1e-7, &beta_init);

    // 5) Final solution: beta = solveA(beta); beta = solveT(beta);
    beta = solveA(beta);
    beta = solveT(beta);

    return beta;
}

// GSC-Falkon
// Vector gscFalkon(const Matrix &Kxm,
//                  const Matrix &Kmm,
//                  const Vector &y,
//                  const Vector &ym,
//                  double lambda,
//                  int n, int t,
//                  double mu0, double q,
//                  int maxNewtonIters,
//                  std::function<double(double,double)> secondDerivLoss)
// {
//     int m = Kmm.rows;
//     Vector alpha(m);
//     for(int i = 0; i < m; i++){
//         alpha[i] = 0.0;
//     }

//     double mu = mu0;
//     for(int k = 0; k < maxNewtonIters; k++){
//         alpha = weightedFalkon(Kxm, Kmm, y, ym, mu, n, t, alpha, secondDerivLoss);
//         mu = q * mu;
//         if(mu < lambda){
//             break;
//         }
//     }

//     alpha = weightedFalkon(Kxm, Kmm, y, ym, lambda, n, t , alpha, secondDerivLoss);

//     return alpha;
// }
Vector gscFalkon(const Matrix &Kxm,
                 const Matrix &Kmm,
                 const Vector &y,
                 const Vector &ym,
                 double lambda,
                 int n, int t,
                 double mu0, double q,
                 int maxNewtonIters,
                 std::function<double(double,double)> secondDerivLoss)
{
    int m = Kmm.rows;
    Vector alpha(m);
    // Parallelize the initialization of alpha to 0 (if m is large)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++){
        alpha[i] = 0.0;
    }

    double mu = mu0;
    // Newton iterations: these must remain sequential because each iteration uses the previous alpha
    for (int k = 0; k < maxNewtonIters; k++){
        alpha = weightedFalkon(Kxm, Kmm, y, ym, mu, n, t, alpha, secondDerivLoss);
        mu = q * mu;
        if (mu < lambda){
            break;
        }
    }

    // Final call with lambda
    alpha = weightedFalkon(Kxm, Kmm, y, ym, lambda, n, t, alpha, secondDerivLoss);

    return alpha;
}

// std::pair<Vector, Vector> primalandHaufeLinearWeights(const int n, 
//                                                       const int d, 
//                                                       const Vector &alpha, 
//                                                       const Matrix &X) 
// {
//     Vector w(d);
//     Vector p(d);
//     Matrix Cov(d, d);

//     // Compute w: for each feature j, sum over samples i.
//     #pragma omp parallel for schedule(static)
//     for (int j = 0; j < d; j++){
//         double sum_j = 0.0;
//         #pragma omp simd reduction(+:sum_j)
//         for (int i = 0; i < n; i++){
//             sum_j += alpha[i] * X(i, j);
//         }
//         w[j] = sum_j;
//     }

//     // Compute covariance: Cov(r, c) = (1/n) * sum_{i=0}^{n-1} X(i, r) * X(i, c)
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int r = 0; r < d; r++){
//         for (int c = 0; c < d; c++){
//             double sum_rc = 0.0;
//             #pragma omp simd reduction(+:sum_rc)
//             for (int i = 0; i < n; i++){
//                 sum_rc += X(i, r) * X(i, c);
//             }
//             Cov(r, c) = sum_rc / double(n);
//         }
//     }

//     // Compute p: for each feature r, p[r] = sum_{c=0}^{d-1} Cov(r, c) * w[c]
//     #pragma omp parallel for schedule(static)
//     for (int r = 0; r < d; r++){
//         double sum_r = 0.0;
//         #pragma omp simd reduction(+:sum_r)
//         for (int c = 0; c < d; c++){
//             sum_r += Cov(r, c) * w[c];
//         }
//         p[r] = sum_r;
//     }
    
//     return {w, p};
// }
// Computes the weighted sum vector and the covariance matrix times that vector,
// then returns the pair {w, p}, where:
//   w = sum_i (alpha[i] * X(i, :))
//   Cov = (1/n) * sum_i X(i, :)^T * X(i, :)
//   p = Cov * w
std::pair<Vector, Vector> primalandHaufeLinearWeights(const int n, 
                                                      const int d, 
                                                      const Vector &alpha, 
                                                      const Matrix &X) 
{
    // Allocate output vector and matrix.
    Vector w(d);
    Matrix Cov(d, d);
    
    // We'll accumulate w and Cov by iterating over samples (rows) because
    // each row X(i,:) is stored contiguously.
    // Create thread–local accumulators to reduce contention.
    int numThreads = omp_get_max_threads();
    // std::vector<std::vector<double>> w_local(numThreads, std::vector<double>(d, 0.0));
    Matrix w_local(numThreads, d);  // each row will correspond to one thread's accumulator
    std::vector<Matrix> Cov_local(numThreads, Matrix(d, d)); // each thread gets its own dxd matrix.

    #pragma omp parallel for schedule(static) //num_threads(26)
    for (int i = 0; i < n; i++){
        int tid = omp_get_thread_num();
        // For sample i, X(i,:) is stored contiguously.
        // Accumulate contribution to w: w_local[tid][j] += alpha[i] * X(i, j)
        double a = alpha[i];
        for (int j = 0; j < d; j++){
            w_local(tid, j) += a * X(i, j);
        }
        
        // Accumulate contribution to covariance matrix:
        // For each pair (r, c), add X(i, r)*X(i, c)
        for (int r = 0; r < d; r++){
            double X_ir = X(i, r);
            for (int c = 0; c < d; c++){
                Cov_local[tid](r, c) += X_ir * X(i, c);
            }
        }
    }

    // Reduce thread–local accumulators into global w and Cov.
    for (int tid = 0; tid < numThreads; tid++){
        for (int j = 0; j < d; j++){
            w[j] += w_local(tid, j);
        }
        for (int r = 0; r < d; r++){
            for (int c = 0; c < d; c++){
                Cov(r, c) += Cov_local[tid](r, c);
            }
        }
    }
    
    // Scale Cov by 1/n.
    for (int r = 0; r < d; r++){
        for (int c = 0; c < d; c++){
            Cov(r, c) /= double(n);
        }
    }
    
    // Now compute p = Cov * w.
    Vector p(d);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < d; r++){
        double sum = 0.0;
        for (int c = 0; c < d; c++){
            sum += Cov(r, c) * w[c];
        }
        p[r] = sum;
    }
    
    return {w, p};
}

// Assume Matrix and Vector are defined as before, with Matrix storing data contiguously.
// Not ideal
// std::pair<Vector, Vector> primalandHaufeLinearWeights(const int n, 
//                                                       const int d, 
//                                                       const Vector &alpha, 
//                                                       const Matrix &X) 
// {
//     // Ensure that the dimensions passed match the dimensions in X.
//     assert(n == X.rows && "n must equal X.rows");
//     assert(d == X.cols && "d must equal X.cols");
    
//     Vector w(d);
//     Matrix Cov(d, d);
    
//     // First, compute w = sum_{i=0}^{n-1} alpha[i] * X(i, :)
//     #pragma omp parallel for schedule(static)
//     for (int j = 0; j < d; j++){
//         double sum_j = 0.0;
//         for (int i = 0; i < n; i++){
//             sum_j += alpha[i] * X(i, j);
//         }
//         w[j] = sum_j;
//     }
//     std::cout << "w calculated.\n";
//     std::cout << "Cov: " << Cov.rows << " x " << Cov.cols << "\n";

//     // Process the n samples in blocks to accumulate the covariance matrix.
//     int blockSize = 256;  // Adjust this value as needed.
    
//     // Zero the global covariance matrix.
//     for (int r = 0; r < d; r++){
//         for (int c = 0; c < d; c++){
//             Cov(r, c) = 0.0;
//         }
//     }

//     std::cout << "cov calculated.\n";
//     // Process samples in blocks.
//     for (int b = 0; b < n; b += blockSize) {
//         int blockEnd = std::min(n, b + blockSize);
//         Matrix Cov_block(d, d);  // temporary block accumulator (zero-initialized)
        
//         // For each sample in the block, accumulate X(i,:) * X(i,:)^T into Cov_block.
//         #pragma omp parallel for schedule(static)
//         for (int i = b; i < blockEnd; i++){
//             for (int r = 0; r < d; r++){
//                 double X_ir = X(i, r);
//                 for (int c = 0; c < d; c++){
//                     double prod = X_ir * X(i, c);
//                     // Use an atomic update to ensure thread-safety.
//                     #pragma omp atomic
//                     Cov_block(r, c) += prod;
//                 }
//             }
//         }
        
//         // Accumulate the block's covariance into the global covariance matrix.
//         for (int r = 0; r < d; r++){
//             for (int c = 0; c < d; c++){
//                 Cov(r, c) += Cov_block(r, c);
//             }
//         }
//     }
    
//     // Scale Cov by 1/n.
//     for (int r = 0; r < d; r++){
//         for (int c = 0; c < d; c++){
//             Cov(r, c) /= double(n);
//         }
//     }
    
//     // Compute p = Cov * w.
//     Vector p(d);
//     #pragma omp parallel for schedule(static)
//     for (int r = 0; r < d; r++){
//         double sum = 0.0;
//         for (int c = 0; c < d; c++){
//             sum += Cov(r, c) * w[c];
//         }
//         p[r] = sum;
//     }
    
//     return {w, p};
// }

// std::pair<Vector, Vector> primalandHaufeLinearWeights(const int n, 
//                                                       const int d, 
//                                                       const Vector &alpha, 
//                                                       const Matrix &X) 
// {
//     // Global outputs.
//     Vector w(d);
//     Matrix Cov(d, d);
    
//     int numThreads = omp_get_max_threads();
//     // Thread-local accumulators for w: one row per thread.
//     Matrix w_local(numThreads, d);
//     // Thread-local compensation for w.
//     std::vector<Vector> w_comp(numThreads, Vector(d));
    
//     // Thread-local accumulators for Cov: one full dxd matrix per thread.
//     std::vector<Matrix> Cov_local(numThreads, Matrix(d, d));
//     // Thread-local compensation for Cov.
//     std::vector<Matrix> Cov_comp(numThreads, Matrix(d, d));
    
//     // Parallel loop: each thread processes a subset of samples.
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < n; i++){
//         int tid = omp_get_thread_num();
//         double a = alpha[i];
//         // Update thread-local w using Kahan compensation.
//         for (int j = 0; j < d; j++){
//             double prod = a * X(i, j);
//             double y = prod - w_comp[tid][j];
//             double t = w_local(tid, j) + y;
//             w_comp[tid][j] = (t - w_local(tid, j)) - y;
//             w_local(tid, j) = t;
//         }
//         // Update thread-local Cov using Kahan compensation.
//         for (int r = 0; r < d; r++){
//             double X_ir = X(i, r);
//             for (int c = 0; c < d; c++){
//                 double prod = X_ir * X(i, c);
//                 double y = prod - Cov_comp[tid](r, c);
//                 double t = Cov_local[tid](r, c) + y;
//                 Cov_comp[tid](r, c) = (t - Cov_local[tid](r, c)) - y;
//                 Cov_local[tid](r, c) = t;
//             }
//         }
//     }
    
//     // Reduce thread-local w accumulators into global w.
//     for (int tid = 0; tid < numThreads; tid++){
//         for (int j = 0; j < d; j++){
//             w[j] += w_local(tid, j);
//         }
//     }
    
//     // Reduce thread-local Cov accumulators into global Cov.
//     for (int tid = 0; tid < numThreads; tid++){
//         for (int r = 0; r < d; r++){
//             for (int c = 0; c < d; c++){
//                 Cov(r, c) += Cov_local[tid](r, c);
//             }
//         }
//     }
    
//     // Scale Cov by 1/n.
//     for (int r = 0; r < d; r++){
//         for (int c = 0; c < d; c++){
//             Cov(r, c) /= double(n);
//         }
//     }
    
//     // Compute Haufe weights: p = Cov * w.
//     Vector p(d);
//     #pragma omp parallel for schedule(static)
//     for (int r = 0; r < d; r++){
//         double sum = 0.0;
//         for (int c = 0; c < d; c++){
//             sum += Cov(r, c) * w[c];
//         }
//         p[r] = sum;
//     }
    
//     return {w, p};
// }

double logistic_second_deriv(double z, double y) {
    double val = y * z;
    double sig = 1.0 / (1.0 + std::exp(-val));
    return sig * (1.0 - sig);
}

bool loadGaussianData(const std::string &filename, Matrix &M) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << "\n";
        return false;
    }
    
    size_t nSamples, nFeatures;
    infile.read(reinterpret_cast<char*>(&nSamples), sizeof(nSamples));
    infile.read(reinterpret_cast<char*>(&nFeatures), sizeof(nFeatures));
    
    // Allocate the matrix with the correct dimensions.
    M = Matrix((int)nSamples, (int)nFeatures);
    
    // Read all the doubles directly into the contiguous data vector.
    infile.read(reinterpret_cast<char*>(M.data.data()),
                M.data.size() * sizeof(double));
    infile.close();
    return true;
}

int main() {
    Matrix X(0, 0);
    if (!loadGaussianData("gaussian_data_alt.bin", X)) {
        return 1;
    }
    
    int nSamples = X.rows;
    int nFeatures = X.cols;
    
    std::cout << "Loaded dataset with " << nSamples 
              << " samples and " << nFeatures << " features.\n";
    
    // Create random dummy labels (-1 or 1) for each sample.
    std::vector<double> y(nSamples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 1);
    for (int i = 0; i < nSamples; i++) {
        y[i] = (dist(gen) == 0) ? -1.0 : 1.0;
    }
    std::cout << "Created dummy labels.\n";
    
    // Now, if you need to create a subset (e.g., every 5th sample), do it as follows:
    int m = nSamples / 5;
    Matrix Xm(m, nFeatures);
    std::vector<double> ym(m);
    for (int i = 0; i < m; i++) {
        int idx = 5 * i;
        // Copy row 'idx' from X into XmMat.
        for (int j = 0; j < nFeatures; j++) {
            Xm(i, j) = X(idx, j);
        }
        ym[i] = y[idx];
    }
    
    std::cout << "Subset selected: " << m << " samples.\n";

    double gamma = 0.2;
    int blockSize = 64;
    Matrix Kmm = computeKernelMatrixBlocked(Xm, Xm, gamma, blockSize);
    Matrix Kxm = computeKernelMatrixBlocked(X, Xm, gamma, blockSize);

    std::cout << "Kmm/Kxm solved\n";

    double lambda = 5e-1;
    double mu0 = 1e-1;
    double q = 0.5;
    int maxNewtonIters = 50;
    int cgIters = 50;

    int n = nSamples;
    int d = nFeatures;  // nFeatures was loaded earlier
    Vector yv = Vector(y);
    Vector yvm = Vector(ym);
    
    auto alpha = gscFalkon(Kxm, Kmm,
                           yv, yvm,
                           lambda, n, cgIters,
                           mu0, q, maxNewtonIters,
                           logistic_second_deriv);

    std::cout << "Final alpha:\n";
    for(int i = 0; i < m; i++){
        std::cout << alpha[i] << " ";
    }
    std::cout << "\n\n";

    auto [primalLinearWeights, haufeLinearWeights] = primalandHaufeLinearWeights(n, d, alpha, X);

    std::cout << "Final primal linear weights:\n";
    for(int i = 0; i < m; i++){
        std::cout << primalLinearWeights[i] << " ";
    }
    std::cout << "\n";

    // std::cout << "Final Haufe linear weights:\n";
    // for(int i = 0; i < m; i++){
    //     std::cout << haufeLinearWeights[i] << " ";
    // }
    // std::cout << "\n";

    return 0;
}