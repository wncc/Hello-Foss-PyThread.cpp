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
    // Initialize a simple lock for parallel region
    omp_lock_t lock;
    omp_init_lock(&lock);

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
                omp_set_lock(&lock);
                u[k][j] = a[k][j];
                for (int s = 0; s < k; s++) {
                    u[k][j] -= l[k][s] * u[s][j];
                }
                omp_unset_lock(&lock);
            }

            // Update L matrix
            #pragma omp for schedule(static)
            for (int i = k + 1; i < size; i++) {
                omp_set_lock(&lock);
                l[i][k] = a[i][k];
                for (int s = 0; s < k; s++) {
                    l[i][k] -= l[i][s] * u[s][k];
                }
                l[i][k] /= u[k][k];
                omp_unset_lock(&lock);
            }
        }
    }

    omp_destroy_lock(&lock);
}
int main(int argc, char *argv[]) {
    int size;
    cout<<"Please Enter size"<<endl;
    cin>>size;//accept the size of the array
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
    //Instead of manuall setting the value of an array we allow the user to input array values
    cout << "Enter the elements of the matrix A:" << endl;
    for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
        cin >> a[i][j];
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
    //Free the dynamicall allocated memory 
    for (int i = 0; i < size; i++) {
        free(a[i]);
        free(l[i]);
        free(u[i]);
    }
    free(a);
    free(l);
    free(u);

    return 0;
}