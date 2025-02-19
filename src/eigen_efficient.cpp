// // #include <Eigen/Dense>
// // #include <Eigen/SVD>
// // #include <iostream>
// // #include <utility>
// // #include <random>
// // #include <cstdlib>
// // #include <vector>
// // #include <numeric>
// // #include <algorithm>
// // #include <tuple>

// // using namespace Eigen;
// // using namespace std;

// // /// Compute the dual (alpha), primal (w), and Haufe weights (p) using an SVD-based approach.
// // /// Given:
// // ///   X (n x d): data matrix (each row is a sample)
// // ///   y (n x 1): target vector
// // ///   lambda: regularization parameter
// // /// This function computes:
// // ///   alpha = U * diag(1/(sigma^2 + lambda)) * U^T * y
// // ///   w     = X^T * alpha         (primal weights for prediction)
// // ///   p     = (1/n) * X^T * (X * w)   (Haufe weights for interpretability)
// // /// Returns a tuple (alpha, w, p).
// // tuple<VectorXd, VectorXd, VectorXd> computeAlphaAndHaufeWeights(const MatrixXd &X,
// //                                                                   const VectorXd &y,
// //                                                                   double lambda,
// //                                                                   double svdTol = 1e-10) {
// //     const int n = X.rows();
// //     // Compute A = X*X^T (an n x n matrix)
// //     MatrixXd A = X * X.transpose();

// //     // Compute the eigen-decomposition of A.
// //     SelfAdjointEigenSolver<MatrixXd> eigensolver(A);
// //     if (eigensolver.info() != Success) {
// //         cerr << "Eigen decomposition failed!" << endl;
// //         exit(1);
// //     }
// //     VectorXd eigVals = eigensolver.eigenvalues();   // sorted in increasing order
// //     MatrixXd Q = eigensolver.eigenvectors();          // columns are eigenvectors

// //     // Build the inverse diagonal using 1/(eig + lambda)
// //     VectorXd invDiag(eigVals.size());
// //     for (int i = 0; i < eigVals.size(); i++) {
// //         if (eigVals(i) > svdTol)
// //             invDiag(i) = 1.0 / (eigVals(i) + lambda);
// //         else
// //             invDiag(i) = 0.0;
// //     }

// //     // Compute alpha = Q * diag(invDiag) * Q^T * y
// //     VectorXd alpha = Q * (invDiag.array() * (Q.transpose() * y).array()).matrix();

// //     // Compute primal weights: w = X^T * alpha.
// //     VectorXd w = X.transpose() * alpha;

// //     // Compute Haufe weights: p = (1/n) * X^T * (X * w)
// //     VectorXd p = (X.transpose() * (X * w)) / double(n);


// //     return make_tuple(alpha, w, p);
// // }

// // double computeMSE(const VectorXd &y_true, const VectorXd &y_pred) {
// //     return (y_true - y_pred).squaredNorm() / y_true.size();
// // }

// // int main(){
// //     // Example parameters:
// //     const int n = 5000;     // Number of samples
// //     const int d = 77000;    // Number of features (use a smaller d for testing if needed)
// //     double lambda = 5e-1;   // Regularization parameter
// //     const int kFolds = 5;   // 5-fold cross validation

// //     // For reproducibility, set a random seed.
// //     srand(42);
// //     static std::mt19937 gen(42);
// //     normal_distribution<double> nd(0.0, 1.0);

// //     // Generate random data X (n x d) and target vector y (n x 1)
// //     MatrixXd X(n, d);
// //     VectorXd y(n);
// //     for (int i = 0; i < n; i++){
// //         for (int j = 0; j < d; j++){
// //             X(i,j) = nd(gen);
// //         }
// //         y(i) = nd(gen);
// //     }

// //     // Create a vector of indices and shuffle them.
// //     vector<int> indices(n);
// //     iota(indices.begin(), indices.end(), 0);
// //     shuffle(indices.begin(), indices.end(), gen);

// //     // Determine fold sizes.
// //     int foldSize = n / kFolds;
// //     vector<double> foldErrors;

// //     // Cross-validation loop:
// //     for (int fold = 0; fold < kFolds; fold++) {
// //         // Determine test indices for the current fold.
// //         int start = fold * foldSize;
// //         int end = (fold == kFolds - 1) ? n : start + foldSize;
// //         vector<int> testIdx(indices.begin() + start, indices.begin() + end);

// //         // Determine training indices (all indices not in testIdx).
// //         vector<int> trainIdx;
// //         for (int idx : indices) {
// //             if (find(testIdx.begin(), testIdx.end(), idx) == testIdx.end())
// //                 trainIdx.push_back(idx);
// //         }

// //         // Build training data matrices.
// //         MatrixXd X_train(trainIdx.size(), d);
// //         VectorXd y_train(trainIdx.size());
// //         for (int i = 0; i < trainIdx.size(); i++) {
// //             X_train.row(i) = X.row(trainIdx[i]);
// //             y_train(i) = y(trainIdx[i]);
// //         }

// //         // Build test data matrices.
// //         MatrixXd X_test(testIdx.size(), d);
// //         VectorXd y_test(testIdx.size());
// //         for (int i = 0; i < testIdx.size(); i++) {
// //             X_test.row(i) = X.row(testIdx[i]);
// //             y_test(i) = y(testIdx[i]);
// //         }

// //         // Train the model on the training set.
// //         auto [alpha, w, p] = computeAlphaAndHaufeWeights(X_train, y_train, lambda);

// //         // Make predictions on the test set: y_pred = X_test * w.
// //         VectorXd y_pred = X_test * w;

// //         // Compute and record the mean squared error (MSE) for this fold.
// //         double mse = computeMSE(y_test, y_pred);
// //         foldErrors.push_back(mse);

// //         cout << "Fold " << fold + 1 << " MSE: " << mse << "\n";
// //     }

// //     // Compute the average MSE over all folds.
// //     double avgMSE = accumulate(foldErrors.begin(), foldErrors.end(), 0.0) / foldErrors.size();
// //     cout << "\nAverage MSE over " << kFolds << " folds: " << avgMSE << "\n";

// //     return 0;
// // }

// #include <Eigen/Dense>
// #include <Eigen/Cholesky>
// #include <iostream>
// #include <random>
// #include <cstdlib>
// #include <vector>
// #include <numeric>
// #include <algorithm>
// #include <tuple>

// using namespace Eigen;
// using namespace std;

// // Optimized function using Cholesky decomposition to solve
// //   (X*X^T + lambda I) alpha = y,
// // then computes the primal weights (w = X^T alpha) and Haufe weights.
// tuple<VectorXd, VectorXd, VectorXd> computeAlphaAndHaufeWeightsCholesky(const MatrixXd &X,
//                                                                           const VectorXd &y,
//                                                                           double lambda) {
//     const int n = X.rows();
//     // Compute A = X * X^T + lambda I.
//     MatrixXd A = X * X.transpose();
//     A.diagonal().array() += lambda;
//     // Solve A * alpha = y using LLT (Cholesky factorization).
//     VectorXd alpha = A.llt().solve(y);
//     // Compute primal weights: w = X^T * alpha.
//     VectorXd w = X.transpose() * alpha;
//     // Compute Haufe weights: p = (1/n) * X^T * (X * w)
//     VectorXd p = (X.transpose() * (X * w)) / double(n);
//     return make_tuple(alpha, w, p);
// }

// double computeMSE(const VectorXd &y_true, const VectorXd &y_pred) {
//     return (y_true - y_pred).squaredNorm() / y_true.size();
// }

// int main(){
//     // Example parameters:
//     const int n = 1000;     // Number of samples
//     const int d = 77000;    // Number of features
//     double lambda = 5e-1;   // Regularization parameter
//     const int kFolds = 5;   // 5-fold cross validation

//     // For reproducibility, set up a random generator.
//     std::mt19937 gen(42);
//     normal_distribution<double> nd(0.0, 1.0);

//     // Generate random data X (n x d) and target vector y (n x 1).
//     MatrixXd X(n, d);
//     VectorXd y(n);
//     for (int i = 0; i < n; i++){
//         for (int j = 0; j < d; j++){
//             X(i, j) = nd(gen);
//         }
//         y(i) = nd(gen);
//     }

//     // Create a permutation of indices for cross validation.
//     vector<int> indices(n);
//     iota(indices.begin(), indices.end(), 0);
//     shuffle(indices.begin(), indices.end(), gen);

//     // Build an Eigen permutation matrix from the shuffled indices.
//     PermutationMatrix<Dynamic, Dynamic, int> perm(n);
//     for (int i = 0; i < n; i++){
//         perm.indices()(i) = indices[i];
//     }
//     // Permute the rows of X and y.
//     MatrixXd X_shuffled = perm * X;
//     VectorXd y_shuffled = perm * y;

//     int foldSize = n / kFolds;
//     vector<double> foldErrors;

//     // Cross-validation loop.
//     for (int fold = 0; fold < kFolds; fold++){
//         int start = fold * foldSize;
//         int testSize = (fold == kFolds - 1) ? (n - start) : foldSize;

//         // Extract test set as a contiguous block.
//         MatrixXd X_test = X_shuffled.middleRows(start, testSize);
//         VectorXd y_test = y_shuffled.segment(start, testSize);

//         // Build training set by concatenating the parts before and after the test block.
//         int trainSize = n - testSize;
//         MatrixXd X_train(trainSize, d);
//         VectorXd y_train(trainSize);
//         if (start > 0) {
//             X_train.topRows(start) = X_shuffled.topRows(start);
//             y_train.head(start) = y_shuffled.head(start);
//         }
//         if (n - start - testSize > 0) {
//             X_train.bottomRows(n - start - testSize) = X_shuffled.bottomRows(n - start - testSize);
//             y_train.tail(n - start - testSize) = y_shuffled.tail(n - start - testSize);
//         }

//         // Train the model on the training set using the optimized (Cholesky-based) function.
//         auto [alpha, w, p] = computeAlphaAndHaufeWeightsCholesky(X_train, y_train, lambda);
//         // auto [alpha, w, p] = computeAlphaAndHaufeWeights(X_train, y_train, lambda);
//         // Predict on the test set: y_pred = X_test * w.
//         VectorXd y_pred = X_test * w;
//         double mse = computeMSE(y_test, y_pred);
//         foldErrors.push_back(mse);
//         cout << "Fold " << fold + 1 << " MSE: " << mse << "\n";
//     }

//     // Compute and print the average MSE over all folds.
//     double avgMSE = accumulate(foldErrors.begin(), foldErrors.end(), 0.0) / foldErrors.size();
//     cout << "\nAverage MSE over " << kFolds << " folds: " << avgMSE << "\n";

//     return 0;
// }

// #include <Eigen/Dense>
// #include <Eigen/Cholesky>
// #include <iostream>
// #include <random>
// #include <cstdlib>
// #include <vector>
// #include <numeric>
// #include <algorithm>
// #include <tuple>

// using namespace Eigen;
// using namespace std;

// // Optimized function using Cholesky decomposition to solve:
// //    (X*X^T + lambda I) alpha = y,
// // then computes the primal weights (w = X^T * alpha) and Haufe weights.
// tuple<VectorXd, VectorXd, VectorXd> computeAlphaAndHaufeWeightsCholesky(const MatrixXd &X,
//                                                                           const VectorXd &y,
//                                                                           double lambda) {
//     const int n = X.rows();
//     // Form A = X*X^T + lambda*I.
//     MatrixXd A = X * X.transpose();
//     A.diagonal().array() += lambda;
//     // Solve for alpha via Cholesky factorization.
//     VectorXd alpha = A.llt().solve(y);
//     // Compute the primal weights: w = X^T * alpha.
//     VectorXd w = X.transpose() * alpha;
//     // Compute Haufe weights: p = (1/n) * X^T * (X * w).
//     VectorXd p = (X.transpose() * (X * w)) / static_cast<double>(n);
//     return make_tuple(alpha, w, p);
// }

// double computeMSE(const VectorXd &y_true, const VectorXd &y_pred) {
//     return (y_true - y_pred).squaredNorm() / y_true.size();
// }

// int main(){
//     // Full dataset parameters.
//     const int fullN = 5704; // Full sample size.
//     const int d = 77000;    // Number of features.
//     double lambda = 5e-1;   // Regularization parameter.

//     // Define the training subsample sizes.
//     vector<int> trainingSizes = {25, 33, 50, 70, 100, 135, 135, 200, 265, 375, 525, 725, 1000, 1430, 2000, 2800, 3928};

//     // Set up random number generation.
//     std::mt19937 gen(42);
//     normal_distribution<double> nd(0.0, 1.0);

//     // Generate the full dataset X (fullN x d) and target vector y (fullN x 1).
//     MatrixXd X(fullN, d);
//     VectorXd y(fullN);
//     for (int i = 0; i < fullN; i++){
//         for (int j = 0; j < d; j++){
//             X(i, j) = nd(gen);
//         }
//         y(i) = nd(gen);
//     }

//     // For each training subsample size, randomly select training data,
//     // use the rest as test data, train the model, and compute test MSE.
//     cout << "Training subsample size, Test set size, MSE" << "\n";
//     for (int tSize : trainingSizes) {
//         if(tSize >= fullN) {
//             cerr << "Training size (" << tSize << ") must be less than full sample size (" << fullN << ")." << "\n";
//             continue;
//         }
//         // Create a vector of full indices and shuffle.
//         vector<int> indices(fullN);
//         iota(indices.begin(), indices.end(), 0);
//         shuffle(indices.begin(), indices.end(), gen);

//         // Select the first tSize indices for training.
//         vector<int> trainIdx(indices.begin(), indices.begin() + tSize);
//         // The remaining indices form the test set.
//         vector<int> testIdx(indices.begin() + tSize, indices.end());

//         // Build the training dataset.
//         MatrixXd X_train(tSize, d);
//         VectorXd y_train(tSize);
//         for (int i = 0; i < tSize; i++){
//             X_train.row(i) = X.row(trainIdx[i]);
//             y_train(i) = y(trainIdx[i]);
//         }
//         // Build the test dataset.
//         int testSize = fullN - tSize;
//         MatrixXd X_test(testSize, d);
//         VectorXd y_test(testSize);
//         for (int i = 0; i < testSize; i++){
//             X_test.row(i) = X.row(testIdx[i]);
//             y_test(i) = y(testIdx[i]);
//         }

//         // Train the model on the training set.
//         auto [alpha, w, p] = computeAlphaAndHaufeWeightsCholesky(X_train, y_train, lambda);
//         // Make predictions on the test set using the primal weights.
//         VectorXd y_pred = X_test * w;
//         double mse = computeMSE(y_test, y_pred);
//         cout << tSize << ", " << testSize << ", " << mse << "\n";
//     }

//     return 0;
// }

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <iostream>
#include <random>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>

using namespace Eigen;
using namespace std;

// Optimized function using Cholesky decomposition to solve:
//    (X*X^T + lambda I) alpha = y,
// then computes the primal weights (w = X^T * alpha) and Haufe weights.
tuple<VectorXd, VectorXd, VectorXd> computeAlphaAndHaufeWeightsCholesky(const MatrixXd &X,
                                                                          const VectorXd &y,
                                                                          double lambda) {
    const int n = X.rows();
    // Form A = X * X^T + lambda * I.
    MatrixXd A = X * X.transpose();
    A.diagonal().array() += lambda;
    // Solve for alpha using Cholesky factorization.
    VectorXd alpha = A.llt().solve(y);
    // Compute primal weights: w = X^T * alpha.
    VectorXd w = X.transpose() * alpha;
    // Compute Haufe weights: p = (1/n) * X^T * (X * w).
    VectorXd p = (X.transpose() * (X * w)) / static_cast<double>(n);
    return make_tuple(alpha, w, p);
}

double computeMSE(const VectorXd &y_true, const VectorXd &y_pred) {
    return (y_true - y_pred).squaredNorm() / y_true.size();
}

int main(){
    // Full dataset parameters.
    const int fullN = 5704; // Full sample size.
    const int d = 77000;    // Number of features.
    double lambda = 5e-1;   // Regularization parameter.

    // Desired training subsample sizes.
    vector<int> trainingSizes = {25, 33, 50, 70, 100, 135, 135, 200, 265, 375, 525, 725, 1000, 1430, 2000, 2800, 3928};

    // Set up random number generation.
    std::mt19937 gen(42);
    normal_distribution<double> nd(0.0, 1.0);

    // Generate full dataset X (fullN x d) and target vector y (fullN x 1).
    MatrixXd X(fullN, d);
    VectorXd y(fullN);
    for (int i = 0; i < fullN; i++){
        for (int j = 0; j < d; j++){
            X(i, j) = nd(gen);
        }
        y(i) = nd(gen);
    }

    // Create a vector of indices and shuffle them.
    vector<int> indices(fullN);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);
    
    // Build an Eigen permutation matrix from the shuffled indices.
    PermutationMatrix<Dynamic, Dynamic, int> perm(fullN);
    for (int i = 0; i < fullN; i++){
        perm.indices()(i) = indices[i];
    }
    
    // Permute the rows of X and y once.
    MatrixXd X_shuffled = perm * X;
    VectorXd y_shuffled = perm * y;

    cout << "TrainingSize, TestSize, MSE" << "\n";
    for (int tSize : trainingSizes) {
        if(tSize >= fullN) {
            cerr << "Training size (" << tSize << ") must be less than full sample size (" << fullN << ")." << "\n";
            continue;
        }
        // Use topRows and tail to split the dataset into training and test sets.
        MatrixXd X_train = X_shuffled.topRows(tSize);
        VectorXd y_train = y_shuffled.head(tSize);
        int testSize = fullN - tSize;
        MatrixXd X_test = X_shuffled.bottomRows(testSize);
        VectorXd y_test = y_shuffled.tail(testSize);

        // Train the model on the training set.
        auto [alpha, w, p] = computeAlphaAndHaufeWeightsCholesky(X_train, y_train, lambda);

        // Predict on the test set using the primal weights: y_pred = X_test * w.
        VectorXd y_pred = X_test * w;
        double mse = computeMSE(y_test, y_pred);
        cout << tSize << ", " << testSize << ", " << mse << "\n";
    }

    return 0;
}
