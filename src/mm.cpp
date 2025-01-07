#include "../include/matrix_mult.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Define matrices A, B, and C as global variables
int A[N][N];
int B[N][N];
int C[N][N];

int min(int a,int b){
    return (a<b) ?  a : b;
}

// Function to initialize matrices A and B
void initializeMatrices() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 2;
            B[i][j] = 2;
            C[i][j] = 0;  // Initialize C to 0
        }
    }
}

// Function to perform matrix multiplication in parallel
// void matrixMultiply() {
//     int i,j,k;
//     #pragma omp parallel for private( i,  j,  k)
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             for (int k = 0; k < N; k++) {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
// }

void matrixMultiply(){
    const int BLOCK_SIZE = 32;  // size of each block

    // collapse(2) combines the outer 2 loops into a single loop
    // schedule(dynamic,1) allows dynamic scheduling of iterations
    #pragma omp parallel for collapse(2) schedule(dynamic,1)
    for(int ii=0; ii<N; ii+=BLOCK_SIZE){
        for(int jj=0; jj<N; jj+=BLOCK_SIZE){
            for(int kk=0; kk<N; kk+=BLOCK_SIZE){
                // processing starts here

                for(int i=ii; i<min(N,ii+BLOCK_SIZE); i++){
                    for(int j=jj; j<min(N,jj+BLOCK_SIZE); j++){
                        int sum =C[i][j];
                        //this vectorises the inner loop
                        // reduction(+:sum) to ensure crrect computation across threads
                        #pragma omp simd reduction(+:sum)
                        for (int k = kk; k < min(kk+BLOCK_SIZE,N); k++) {
                            sum += A[i][k]*B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

// Function to print matrix C
void printResult() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", C[i][j]);
        }
        printf("\n");
    }
}

// Function to calculate and return elapsed time
double calculateElapsedTime(struct timeval tv1, struct timeval tv2) {
    return (double)(tv2.tv_sec - tv1.tv_sec) + (double)(tv2.tv_usec - tv1.tv_usec) * 1.e-6;
}

int main() {
    // Declare time structs for timing
    struct timeval tv1, tv2;
    struct timezone tz;
    double elapsed;

    // Initialize matrices A and B
    initializeMatrices();

    // Get start time
    gettimeofday(&tv1, &tz);

    // Perform matrix multiplication
    matrixMultiply();

    // Get end time
    gettimeofday(&tv2, &tz);

    // Calculate and print elapsed time
    elapsed = calculateElapsedTime(tv1, tv2);
    printf("Elapsed time = %f seconds.\n", elapsed);

    // Optionally, print result matrix C
    //printResult();

    return 0;
}
