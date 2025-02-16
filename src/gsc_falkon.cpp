#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>
#include <utility>
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

Vector operator*(double scalar, const Vector &v) {
    Vector out(v.size);
    for(int i = 0; i < v.size; i++){
        out[i] = scalar * v[i];
    }
    return out;
}

Vector operator+(const Vector &a, const Vector &b) {
    assert(a.size == b.size);
    Vector out(a.size);
    for(int i = 0; i < a.size; i++){
        out[i] = a[i] + b[i];
    }
    return out;
}

Vector operator-(const Vector &a, const Vector &b) {
    assert(a.size == b.size);
    Vector out(a.size);
    for(int i = 0; i < a.size; i++){
        out[i] = a[i] - b[i];
    }
    return out;
}

double dot(const Vector &a, const Vector &b) {
    assert(a.size == b.size);
    double sum = 0.0;
    for(int i = 0; i < a.size; i++){
        sum += a[i] * b[i];
    }
    return sum;
}


// RBF Gaussian kernel
double rbf_kernel(const std::vector<double> &x,
                  const std::vector<double> &z,
                  double gamma) {
    double sumSq = 0.0;
    assert(x.size() == z.size());
    for(int i = 0; i < (int)x.size(); i++){
        double diff = x[i] - z[i];
        sumSq += diff * diff;
    }
    return std::exp(-gamma * sumSq);
}

Matrix computeKernelMatrix(const std::vector<std::vector<double>> &X1,
                           const std::vector<std::vector<double>> &X2,
                           double gamma) {
    Matrix K(X1.size(), X2.size());
    for(int i = 0; i < (int)X1.size(); i++){
        for(int j = 0; j < (int)X2.size(); j++){
            K(i, j) = rbf_kernel(X1[i], X2[j], gamma);
        }
    }
    return K;
}

bool cholesky_inplace(Matrix &M) {
    assert(M.rows == M.cols);
    int n = M.rows;
    for(int k = 0; k < n; k++){
        if(M(k, k) <= 0.0) return false; // check if positive definite

        M(k,k) = std::sqrt(M(k,k));
        for(int i = k+1; i < n; i++){
            M(i, k) /= M(k, k);
        }
        for(int j = k+1; j < n; j++){
            for(int i = j; i < n; i++){
                M(i, j) -= M(i, k) * M(j, k);
            }
        }
    }

    // set upper part to 0s
    for(int r = 0; r <n; r++){
        for(int c = r+1; c < n; c++){
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
Preconditioner weightedPreconditioner(const Matrix &Kmm,
                                      const Vector &ym,
                                      const Vector &alpha,
                                      double lambda,
                                      std::function<double(double, double)> secondDerivLoss)
{
    int m = Kmm.rows;
    // 1) z = Kmm * alpha
    Vector z(m);
    for(int i = 0; i < m; i++){
        double sum = 0.0;
        for(int j = 0; j < m; j++){
            sum += Kmm(i, j) * alpha[j];
        }
        z[i] = sum;
    }
    // 2) T = chol(kmm) (copy Kmm first)
    Matrix Tmat = Kmm;
    bool ok = cholesky_inplace(Tmat);
    if(!ok) std::cerr << "Cholesky failed in WeightedPreconditioner.\n";

    // 3) Build D from second derivatives
    Vector diagD(m);
    for(int i = 0; i < m; i++){
        diagD[i] = secondDerivLoss(z[i], ym[i]);
    }

    // 4) Form (1/m)*T * D * T^T
    Matrix M = Matrix(m, m);
    // M = (1/m)*(T*D*T^T)
    // T is lower-tri => for each col, we scale by D, then do T^T?
    // We will do a naive triple loop for clarity:
    // Let L = T, we want L * D * L^T => M
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            double sum = 0.0;
            for(int k = 0; k < m; k++){
                // D is diagonal => D(k,k) = diagD[k]
                sum += Tmat(i, k) * diagD[k] * Tmat(j, k);
            }
            M(i, j) = (1.0/m) * sum;
        }
    }

    for(int i = 0; i < m; i++){
        M(i, i) += lambda;
    }

    Matrix Amat = M;
    ok = cholesky_inplace(Amat);
    if(!ok) std::cerr << "Cholesky failed in WeightedPreconditioner A.\n";

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
    Preconditioner prec = weightedPreconditioner(Kmm, ym, alpha0, lambda, secondDerivLoss);
    const Matrix &T = prec.T;
    const Matrix &A = prec.A;
    int m = Kmm.rows;

    // 2) Define LinOp(beta)
    // LinOp(beta) = A^(-T)* (K(Xm,X)* D * K(X,Xm)* T^(-1)* A^(-1)* beta) + lambda n * (A^(-1)*beta)

    auto solveA = [&](const Vector &b) {
        // Solve A * x = b => x = forward_sub(A, b), then backward_sub
        Vector x = forward_sub(A, b);
        x = backward_sub(A, x);
        return x; 
    };

    auto solveAT = [&](const Vector &b) {
        // Solve A^T * x = b => use backward_sub then forward_sub w.r.t. A
        // Because A is lower-triangular
        // => x = forward_sub(A^T, b) but A^T is upper => ...
        // simpler to do x = backward_sub(A, b) then forward_sub(A, x)
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

    // Precompute Kxm^T for convenience => Kmx
    Matrix Kmx(Kxm.cols, Kxm.rows);
    for(int i = 0; i < Kxm.rows; i++){
        for(int j = 0; j < Kxm.cols; j++){
            Kmx(j, i) = Kxm(i, j);
        }
    }

    auto LinOp = [&](const Vector &beta) -> Vector {
        // Step 1) v = A^(-1)* beta
        Vector v = solveA(beta);

        // Step 2) z = Kxm * beta => predictions on full dataset (size n)
        // Kxm is n x m, beta is m
        Vector z(n);
        for(int i = 0; i < n; i++){
            double sum = 0.0;
            for(int j = 0; j < m; j++){
                sum += Kxm(i, j) * beta[j];
            }
            z[i] = sum;
        }

        // Step 3) Build D = diag(secondDerivLoss(z[i],y[i])), i=1..n
        // Then mulitply Kmx * d * z or so.
        // c = Kmx * (D * z)
        // => let w = D*z => w[i] = D[i]*z[i]
        Vector w(n);
        for(int i = 0; i < n; i++){
            double sd = secondDerivLoss(z[i], y[i]);
            w[i] = sd * z[i];
        }

        // c = Kmx * w (Kmx is mx n, w is n => c is m)
        Vector c(m);
        for(int i = 0; i < m; i++){
            double sum = 0.0;
            for(int j = 0; j < n; j++){
                sum += Kmx(i, j) * w[j];
            }
            c[i] = sum;
        }

        // c <- T^(-1) * c
        c = solveT(c);

        // c <- A^(-T) * c
        c = solveAT(c);

        // plus lambda n v => total is c + lambda n v
        // we need to do c = c + lambda * n * v
        for(int i = 0; i < m; i++){
            c[i] += (lambda * n) * v[i];
        }
        return c;
    };

    // 3) Build RHS: R = A^(-T) * Kmx * y
    // c = Kmx * y => c in R^m
    Vector c(m);
    for(int i = 0; i < m; i++){
        double sum = 0.0;
        for(int j = 0; j < n; j++){
            sum += Kmx(i, j) * y[j];
        }
        c[i] = sum;
    }

    c = solveT(c);

    Vector R = solveAT(c);

    // 4) Solve LinOp(beta) = R using CG
    Vector beta_init(m);
    for(int i = 0; i < m; i++){
        beta_init[i] = alpha0[i];
    }

    Vector beta = conjugateGradient(LinOp, R, t, 1e-7, &beta_init);

    // 5) final solution
    beta = solveA(beta);
    beta = solveT(beta);

    return beta;
}

// GSC-Falkon
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
    for(int i = 0; i < m; i++){
        alpha[i] = 0.0;
    }

    double mu = mu0;
    for(int k = 0; k < maxNewtonIters; k++){
        alpha = weightedFalkon(Kxm, Kmm, y, ym, mu, n, t, alpha, secondDerivLoss);
        mu = q * mu;
        if(mu < lambda){
            break;
        }
    }

    alpha = weightedFalkon(Kxm, Kmm, y, ym, lambda, n, t , alpha, secondDerivLoss);

    return alpha;
}

std::pair<Vector, Vector> primalandHaufeLinearWeights(const int n, 
                                                     const int d, 
                                                     const Vector &alpha, 
                                                     const std::vector<std::vector<double>> &X) 
{
    Vector w(d);
    Vector p(d);
    Matrix Cov(d, d);

    for (int j = 0; j < d; j++){
        double sum_j = 0.0;
        for (int i = 0; i < n; i++){
            sum_j += alpha[i] * X[i][j];
            // std::cout << alpha[i] << "alpha\n";
            // std::cout << X[i][j] << "X\n";
        }
        w[j] = sum_j;
    }

    for(int r = 0; r < d; r++){
        for(int c = 0; c < d; c++){
            double sum_rc = 0.0;
            for(int i = 0; i < n; i++){
                sum_rc += X[i][r] * X[i][c];
            }
            Cov(r, c) = sum_rc / double(n);
        }
    }

    for(int r = 0; r < d; r++){
        double sum_r = 0.0;
        for(int c = 0; c < d; c++){
            sum_r += Cov(r, c) * w[c];
        }
        p[r] = sum_r;
    }
    return {w, p};
}

double logistic_second_deriv(double z, double y) {
    double val = y * z;
    double sig = 1.0 / (1.0 + std::exp(-val));
    return sig * (1.0 - sig);
}

int main() {
    std::vector<std::vector<double>> X = {
        {0.5, 0.2}, {0.5, 0.6}, {1.0, 1.0},
        {0.9, -0.6}, {1.0, -1.0}, {0.5, -0.9}
    };
    std::vector<double> y = {1, 1, 1, -1, -1, -1};

    int n = (int)X.size();
    int d = 2;

    std::vector<std::vector<double>> Xm = {X[0], X[2], X[4]};
    std::vector<double> ym = {y[0], y[2], y[4]};
    int m = 3;

    double gamma = 0.2;

    Matrix Kmm = computeKernelMatrix(Xm, Xm, gamma); // m x m
    Matrix Kxm = computeKernelMatrix(X, Xm, gamma);  // n x m

    double lambda = 5e-1;
    double mu0 = 1e-1;
    double q = 0.5;
    int maxNewtonIters = 50;
    int cgIters = 20;

    auto alpha = gscFalkon(Kxm, Kmm,
                           Vector(y), Vector(ym),
                           lambda, n, cgIters,
                           mu0, q, maxNewtonIters,
                           logistic_second_deriv);

    std::cout << "Final alpha:\n";
    for(int i = 0; i < m; i++){
        std::cout << alpha[i] << " ";
    }
    std::cout << "\n";

    auto [primalLinearWeights, haufeLinearWeights] = primalandHaufeLinearWeights(n, d, alpha, X);

    std::cout << "Final primal linear weights:\n";
    for(int i = 0; i < m; i++){
        std::cout << primalLinearWeights[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Final Haufe linear weights:\n";
    for(int i = 0; i < m; i++){
        std::cout << haufeLinearWeights[i] << " ";
    }
    std::cout << "\n";


    return 0;
}