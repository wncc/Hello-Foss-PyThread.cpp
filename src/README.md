## Functions Implenented
A list of functions that have been implemented can be found here :-

### 1) LU Factoristion
>This C++ code implements LU factorization using OpenMP for parallel execution of matrix updates. It optimizes the decomposition by distributing computations for the lower (L) and upper (U) triangular matrices across multiple threads.

### 2) Maximum element search
>The code for this function can be found in [max.cpp](src/max.cpp), and input for the following can be found in input.cpp
The code uses OpenMP for parallel programming to find the maximum element in an array. The search is distributed across multiple threads, improving performance by dividing the workload.

### 3) Matrix Matrix Multiplication
>The code for the following function can be found in [mm.cpp](src/mm.cpp)<br>
This code performs matrix-matrix multiplication using OpenMP to parallelize the computation across multiple threads. It optimizes the multiplication process for large matrices, reducing execution time by distributing the workload across available CPU cores.

### 4) Montecarlo Method
>The code for the following function can be found in [montecarlo.cpp](src/montecarlo.cpp)<br>
The code estimates the value of Pi using the Monte Carlo method with OpenMP for parallel processing. It simulates random points within a unit square and counts how many fall within the unit circle, then uses multiple threads to improve performance and speed up the estimation process.

### 5) Matrix Vector Multiplication
>The code for the following function can be found in [mv.cpp](src/mv.cpp)<br>
The code performs matrix-vector multiplication using OpenMP for parallel processing. The dynamic scheduling with a chunk size of 16 distributes the computation of each row of the matrix across multiple threads, optimizing the execution for large-scale data by balancing the load dynamically.

### 6) Product of elements of an array
>The code for the following function can be found in [prod.cpp](src/prod.cpp)<br>
This C++ code calculates the product of elements in an array using OpenMP to parallelize the computation. It optimizes large product calculations by summing the logarithms of array elements in parallel and exponentiating the result to obtain the final product, reducing potential overflow risks.

### 7) Pi reduction
>The code for the following function can be found in [pi-reduction.cpp](src/pi-reduction.cpp)<br>
This C++ code estimates the value of Pi using numerical integration with the OpenMP library for parallelization. It divides the computation of the integral into multiple threads, summing partial results in parallel using a reduction clause to optimize the performance and accuracy when calculating Pi across a large number of steps.

### 8) Calculation of Standard Deviation
>The code for the following function can be found in [standard_dev.cpp](src/standard_dev.cpp)<br>
This C++ code calculates the standard deviation of a dataset using OpenMP for parallel processing. It first computes the mean in parallel, then calculates the variance by summing the squared differences from the mean, distributing both tasks across multiple threads to improve performance with large datasets.

### 9) Sum of elements of an array
>The code for the following function can be found in [sum2.cpp](src/sum2.cpp) <br>
This C++ code computes the sum of a large array (with 10 million elements) in parallel using OpenMP. It divides the workload among multiple threads based on the total number of threads, each thread calculates a partial sum, and the results are combined in a critical section to avoid race conditions. The execution time for the sum computation is also measured and displayed. 

### 10) Vector-Vector Dot product calculation
>The code for the following function can be found in [vvd.cpp](src/vvd.cpp) <br>
This C++ code calculates the dot product of two arrays using OpenMP for parallelization. It initializes two arrays, A and B, each containing 1000 elements set to 1. The dot product is computed in parallel using a dynamic scheduling strategy, with a chunk size of 100, and the results are combined using a reduction operation. The final result is printed to the console.

### 11) Sum calculation (wrong as pragma barrier is not calculated)
>The code for the following function can be found in [wrong_sum.cpp](src/wrong.cpp)<br>
This C++ code computes the sum of an array using OpenMP with task-based parallelism. It initializes an array of size 600 with all elements set to 1. The code divides the summation task into segments of size 100, allowing multiple threads to process these segments concurrently. The results from each task are accumulated into a shared variable sum using a critical section to prevent data races.

## 0.2) Compilation
>
```shell
# compile using g++ for Openmp
g++ sum2.cpp -o sum2 -fopenmp
./sum2

# compile using g++ for MPI
mpic++ -fopenmp sum2.cpp -o sum2
./sum2
```
