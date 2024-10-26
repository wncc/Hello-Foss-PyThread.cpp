// lu_decomposition.cpp
#include "lu_decomposition.h"
#include <omp.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
using namespace std;
// LU Decomposition function
void l_u_d(float** a, float** l, float** u, int size)
{

    // Initialize L and U matrices
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                l[i][j] = 1.0; // Diagonal elements of L are 1
                u[i][j] = a[i][j];
            } else if (i > j) {
                l[i][j] = a[i][j];
                u[i][j] = 0.0;
            } else {
                l[i][j] = 0.0;
                u[i][j] = a[i][j];
            }
        }
    }

    // Parallelize the LU decomposition
    #pragma omp parallel shared(a, l, u)
    {
        for (int k = 0; k < size; k++) {
            // Update U matrix
            #pragma omp for schedule(static)
            for (int j = k; j < size; j++) {
                u[k][j] = a[k][j];
                for (int s = 0; s < k; s++) {
                    u[k][j] -= l[k][s] * u[s][j];
                }
            }

            // Update L matrix
            #pragma omp for schedule(static)
            for (int i = k + 1; i < size; i++) {
                l[i][k] = a[i][k];
                for (int s = 0; s < k; s++) {
                    l[i][k] -= l[i][s] * u[s][k];
                }
                l[i][k] /= u[k][k];
            }
        }
    }

}
int main(int argc, char *argv[]) {
    int size = 2;
    float **a, **l, **u;

    // Allocate memory for the 2D arrays
    a = (float **)malloc(size * sizeof(float *));
    l = (float **)malloc(size * sizeof(float *));
    u = (float **)malloc(size * sizeof(float *));
    for (int i = 0; i < size; i++) {
        a[i] = (float *)malloc(size * sizeof(float));
        l[i] = (float *)malloc(size * sizeof(float));
        u[i] = (float *)malloc(size * sizeof(float));
    }

    // Initialize the array 'a'
    float temp[2][2] = {
        {4, 3},
        {6, 3}
    };
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = temp[i][j];
        }
    }

    // Perform LU decomposition
    l_u_d(a, l, u, size);

    // Print L matrix
    printf("L Matrix:\n");
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            printf("%f ", l[i][j]);
        }
        printf("\n");
    }

    // Print U matrix
    printf("U Matrix:\n");
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            printf("%f ", u[i][j]);
        }
        printf("\n");
    }

    return 0;
}