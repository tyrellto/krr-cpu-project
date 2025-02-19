// #include <iostream>
// #include <vector>
// #include <random>
// #include <fstream>
// #include <omp.h>

// int main() {
//     // Dimensions
//     const size_t nSamples = 5000;
//     const size_t nFeatures = 70000;
//     const size_t totalElements = nSamples * nFeatures;

//     // Allocate contiguous memory for the dataset
//     std::vector<double> data(totalElements);

//     // Set up a random device for seeding
//     std::random_device rd;

//     // Determine the number of threads available
//     int numThreads = omp_get_max_threads();

//     // Create a vector of engines, one per thread
//     std::vector<std::mt19937> engines(numThreads);
//     for (int t = 0; t < numThreads; t++) {
//         engines[t] = std::mt19937(rd() + t);
//     }

//     // Use an explicit parallel region to reduce per-iteration overhead.
//     // Each thread copies its engine to a local variable, and creates its own local distribution.
//     #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         // Copy the engine locally to avoid repeated vector lookups
//         std::mt19937 local_engine = engines[thread_id];
//         // Create a local normal distribution instance
//         std::normal_distribution<double> dist(0.0, 1.0);

//         // Use a static schedule to assign contiguous blocks, improving cache locality.
//         #pragma omp for schedule(static)
//         for (size_t i = 0; i < totalElements; i++) {
//             data[i] = dist(local_engine);
//         }
//     }

//     // (Optional) Verify by printing the first few values
//     std::cout << "First 5 values:" << std::endl;
//     for (int i = 0; i < 5; i++) {
//         std::cout << data[i] << " ";
//     }
//     std::cout << std::endl;

//     // Save the data to a binary file.
//     // The file will first contain nSamples and nFeatures (as size_t values)
//     // followed by the data array.
//     std::ofstream outfile("gaussian_data_alt.bin", std::ios::binary);
//     if (!outfile) {
//         std::cerr << "Error opening file for writing!" << std::endl;
//         return 1;
//     }
    
//     // Write the dimensions so that they can be recovered later.
//     outfile.write(reinterpret_cast<const char*>(&nSamples), sizeof(nSamples));
//     outfile.write(reinterpret_cast<const char*>(&nFeatures), sizeof(nFeatures));
    
//     // Write the raw data array.
//     outfile.write(reinterpret_cast<const char*>(data.data()), totalElements * sizeof(double));
    
//     outfile.close();
//     std::cout << "Data saved to gaussian_data.bin" << std::endl;

//     return 0;
// }
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <omp.h>
#include <cassert>

// Main function.
int main() {
    // Dimensions
    const size_t nSamples = 5000;
    const size_t nFeatures = 7000;
    const size_t totalElements = nSamples * nFeatures;

    // Allocate contiguous memory for the dataset.
    std::vector<double> data(totalElements);
    // Allocate memory for cluster labels (using int; labels will be +1 or -1).
    std::vector<int> labels(nSamples);

    // Set up a random device for seeding.
    std::random_device rd;

    // Determine the number of threads available.
    int numThreads = omp_get_max_threads();

    // Create a vector of random engines, one per thread.
    std::vector<std::mt19937> engines(numThreads);
    for (int t = 0; t < numThreads; t++) {
        engines[t] = std::mt19937(rd() + t);
    }

    // Use an OpenMP parallel region to generate the data and labels.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nSamples; i++) {
        int tid = omp_get_thread_num();
        // Copy the engine locally.
        std::mt19937 local_engine = engines[tid];
        // Create two normal distributions: one for each cluster.
        std::normal_distribution<double> dist_cluster1(2.0, 1.0);
        std::normal_distribution<double> dist_cluster2(-2.0, 1.0);
        
        // Determine cluster membership based on sample index.
        bool isCluster1 = (i < nSamples / 2);
        // Set label: +1 for cluster 1, -1 for cluster 2.
        labels[i] = isCluster1 ? 1 : -1;
        
        // Fill the features for sample i.
        for (size_t j = 0; j < nFeatures; j++) {
            size_t idx = i * nFeatures + j;
            if (isCluster1)
                data[idx] = dist_cluster1(local_engine);
            else
                data[idx] = dist_cluster2(local_engine);
        }
    }

    // (Optional) Print the first 5 samples (first 5 features) with labels.
    std::cout << "First 5 samples:\n";
    for (size_t i = 0; i < 5; i++){
        for (size_t j = 0; j < 5; j++){
            std::cout << data[i * nFeatures + j] << " ";
        }
        std::cout << "Label: " << labels[i] << "\n";
    }

    // Save the data and labels to a binary file.
    std::ofstream outfile("gaussian_data_clusters_alt.bin", std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file for writing!\n";
        return 1;
    }
    
    // Write dimensions (nSamples and nFeatures).
    outfile.write(reinterpret_cast<const char*>(&nSamples), sizeof(nSamples));
    outfile.write(reinterpret_cast<const char*>(&nFeatures), sizeof(nFeatures));
    
    // Write the data array.
    outfile.write(reinterpret_cast<const char*>(data.data()), totalElements * sizeof(double));
    
    // Write the labels array.
    outfile.write(reinterpret_cast<const char*>(labels.data()), nSamples * sizeof(int));
    
    outfile.close();
    std::cout << "Data and labels saved to gaussian_data_clusters.bin\n";

    return 0;
}
