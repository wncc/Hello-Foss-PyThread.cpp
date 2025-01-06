#include "montecarlo.h"
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <chrono>
using namespace std;

unsigned long long getCurrentTimeInMilliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void montecarlo(int n, int num_threads) {
    int pCircle = 0, pSquare = 0;
    double x, y, d;
    int i;
    unsigned long long seed = getCurrentTimeInMilliseconds();
    srand(seed); // Seeding with a modified value

    // Parallelize the loop using OpenMP
    #pragma omp parallel for private(x, y, d, i) reduction(+:pCircle, pSquare) num_threads(num_threads)
    for(i = 0; i < n; i++) {

        // Generate random points between 0 and 1
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;

        // Calculate distance from the origin
        d = x * x + y * y;

        // Check if point is inside the unit circle
        if(d <= 1) {
            pCircle++;
        }
        pSquare++;
    }

    // Calculate and print the approximation of Pi
    double pi = 4 * (double)pCircle / (double)pSquare;
    cout << "Pi: " << pi << endl;
}


int main() {
    int n = 1000000;     // Number of points for the Monte Carlo simulation
    int num_threads = 4; // Number of threads to use

    // Call the Monte Carlo function to estimate Pi
    montecarlo(n, num_threads);

    return 0;
}
