// product_log.cpp
#include "product_log.h"
#include <omp.h>
#include <iostream>
using namespace std;
// Function to initialize the array
void initialize_array(int* a, int size, int value) {
    for (int i = 0; i < size; i++) {
        a[i] = value;
    }
}

// Function to compute the sum of logarithms in parallel
double compute_log_sum(int* a, int size, int num_threads) {
    double log_sum = 0.0;

    #pragma omp parallel for default(shared) reduction(+:log_sum) num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        log_sum += log(abs(a[i]));
    }

    return log_sum;
}

// Function to find number of negative numbers in parallel
pair<int, int> count_negative_and_zero(int* a, int size, int num_threads) {
    int negative_count = 0, zero_count = 0;
    int i;

    #pragma omp parallel for private(i) reduction(+:negative_count, zero_count) num_threads(num_threads)
    for (i = 0; i < size; i++) {
        if(a[i] < 0 )negative_count++;
        else if(a[i] == 0)zero_count++;
    }

    return {negative_count, zero_count};
}

// Function to compute the product using the sum of logarithms
double compute_product(double log_sum) {
    return exp(log_sum);
}



#define ARR_SIZE 100

int main(int argc, char *argv[]) {
    int a[ARR_SIZE];
    int num_threads = 4; // Set the number of threads

    // Initialize array
    initialize_array(a, ARR_SIZE, 2);

    pair<int, int> neg_and_zero_count = count_negative_and_zero(a, ARR_SIZE, num_threads);
    if (neg_and_zero_count.second > 0)
    {
        // Print the result
        cout << "Product=" << 0 << endl;
        return 0;
    }

    // Compute the log sum in parallel
    double log_sum = compute_log_sum(a, ARR_SIZE, num_threads);

    // Compute the product using the log sum
    double prod = compute_product(log_sum);

    if(neg_and_zero_count.first%2 == 1)
    {
        prod = -prod;
    }

    // Print the result
    cout << "Product=" << prod << endl;

    return 0;
}
