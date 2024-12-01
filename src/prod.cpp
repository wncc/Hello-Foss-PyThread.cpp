// product_log.cpp
#include "product_log.h"
#include <omp.h>
#include <iostream>
#include <cmath>
using namespace std;
// Function to initialize the array
void initialize_array(int* a, int size, int value) {
    for (int i = 0; i < size; i++) {
        a[i] = value;
    }
}

// Function to compute the sum of logarithms in parallel (HANDLING NEGATIVE VALUES ALSO)
double compute_log_sum(int* a, int size, int num_threads, bool &zero_val, int &neg_vals) {
    double log_sum = 0.0;

    #pragma omp parallel for default(shared) reduction(+:log_sum) num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        if(a[i] == 0){
            zero_val = true;
        }
        else if(a[i] < 0){
            neg_vals++;
            log_sum += log(abs(a[i]));
        }
        else{
            log_sum += log(a[i]);
        }
    }

    return log_sum;
}


double compute_product(double log_sum, bool &zero_val, int &neg_vals) {
    // Function to compute the product using the sum of logarithms (Returning 0 if any one input is 0)
    if(zero_val){
        return 0.0;
    }
    // If the number of negative values is odd, the product should be negative
    else if((neg_vals%2)!=0){
        return -1.0*exp(log_sum);
    }
    return exp(log_sum);
}



#define ARR_SIZE 100

int main(int argc, char *argv[]) {
    int a[ARR_SIZE];
    int num_threads = 4; // Set the number of threads

    // Initialize array
    initialize_array(a, ARR_SIZE, 2);

    bool zero_val = false; // Initially we don't know if 0 is present in input or no, so we set it to false
    int neg_vals = 0; // Initially number of negative values in input be set as 0

    // Compute the log sum in parallel
    double log_sum = compute_log_sum(a, ARR_SIZE, num_threads, zero_val, neg_vals);

    // Compute the product using the log sum
    double prod = compute_product(log_sum, zero_val, neg_vals);

    // Print the result
    cout << "Product=" << prod << endl;

    return 0;
}
