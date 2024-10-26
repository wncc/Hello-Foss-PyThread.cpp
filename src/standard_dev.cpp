// statistics.cpp
#include "statistics.h"
#include <omp.h>
#include <cmath>
#include <iostream>
// Function to compute the mean
double computeMean(const std::vector<int>& data, int num_threads) {
    double sum = 0.0;

    int num_threads = omp_get_max_threads();
    #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }

    return sum / data.size();
}

// Function to compute the standard deviation
double computeStandardDeviation(const std::vector<int>& data, int num_threads) {
    double mean = computeMean(data, num_threads);

    double variance_sum = 0.0;

    int num_threads = omp_get_max_threads();
    #pragma omp parallel for reduction(+:variance_sum) num_threads(num_threads)
    for (size_t i = 0; i < data.size(); i++) {
        variance_sum += (data[i] - mean) * (data[i] - mean);
    }

    double variance = variance_sum / data.size();
    return std::sqrt(variance);
}


int main() {
    // Data set
    std::vector<int> data = {1, 2, 3, 4, 5, 6};

    // Number of threads for OpenMP
    int num_threads = 4;

    // Calculate and print the standard deviation
    double stddev = computeStandardDeviation(data, num_threads);
    std::cout << "Standard Deviation: " << stddev << std::endl;

    return 0;
}
