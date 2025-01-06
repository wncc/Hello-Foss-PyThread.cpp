# Parallel_Programming
Parallel programming is often used interchangeably with parallel processing or in connection with parallel computing, which refers to systems that enhance the efficiency of parallel execution. In parallel programming, tasks are divided and executed simultaneously across multiple computers or multiple CPU cores. This approach is essential for large-scale projects where speed and precision are crucial. Although complex, parallel programming enables developers, researchers, and users to perform research and analysis much faster than programs limited to processing tasks sequentially

## Project Description
This project demonstrates the use of parallel programming techniques in C++ with (`OpenMP`) and (`MPI`) to efficiently execute computations across multiple threads. It includes examples of parallel loops, task scheduling, and thread synchronization. The goal is to showcase performance improvements in computational tasks using OpenMP and MPI directives.

## **1. OpenMP (Open Multi-Processing)** <a id="openmp"></a>

`OpenMP` is a multi-platform programming interface that enables multiprocessing programming. `OpenMP` can be used in C++, C and Fortran languages, including different architectures like Windows and Unix. It consists of compilator directives that have an impact on code execution.

The `OpenMP` interface is a component of the GNU Compiler Collection (`GCC`), a set of open-source compilers developed by the GNU Project. `GCC` compiler is therefore highly recommended for use with `OpenMP`, although it is not required (there is an Intel compiler that also support `OpenMP`).

**INSTALLATION AND CONFIGURATION ON LINUX SYSTEMS:**

Start the terminal and update the repository:

```bash
>>> sudo apt-get update
```

Then install the `build-essential` package, including `gcc`, `g++` and `make`:

```bash
>>> sudo apt-get install build-essential
```

## **2. MPI** <a id="mpi"></a>

Message Passing Interface (`MPI`) is a communication protocol standard for transferring messages between parallel program processes on one or more computers. `MPI` is currently the most widely used communication model in clusters of computers and supercomputers.

There are several implementations of `MPI`, including `OpenMPI`, `MPICH` and `MSMPI`. On Linux, we can choose from `OpenMPI` and `MPICH`, while `MSMPI` is a Windows implementation. Before going any further, we should ensure that we have the `GCC` compiler installed.

**INSTALLATION AND CONFIGURATION OF `MPICH` ON LINUX SYSTEMS:**

Start the terminal and update the repository:

```bash
>>> sudo apt-get update
```

We then install the `mpich` package:

```bash
>>> sudo apt-get install mpich
```

We can now check the version of the installed `MPI` (this will actually be the `GCC` version):

```bash
>>> mpic++ --version
```

Here you can find out more about `MPICH`: [https://www.mpich.org/](https://www.mpich.org/).


**THE INSTALLATION PROCESS UNDER WINDOWS IS COMPLEX, AND I DO NOT RECOMMEND USING MPI WITH THE WINDOWS PLATFORM**

**NOTE**
> Linux is often recommended for compiling parallel programs
> For optimal performance and seamless development, we recommend using Linux for building and running this project.

## Project Tree
> This project contains two branches: `main` and `Advanced`.  
> The `main` branch includes operations from the NumPy module that have been translated into C++ using OpenMP and MPI for parallel processing.

### **main**

    .
    |-- include
    |   |-- array_concat.h
    |   |-- dot_product.h
    |   |-- find_max.h
    |   |-- lu_decomposition.h
    |   |-- matrix_mult.h
    |   |-- matrix_vector.h
    |   |-- montecarlo.h
    |   |-- parallel_sum.h
    |   |-- pi_calculator.h
    |   |-- product_log.h
    |   |-- statistics.h
    |   |-- sum2.h
    |   |-- sum_task.h
    |
    |-- src
    |   |-- Histogram/
    |   |   |-- histo.cpp
    |   |   |-- data.txt
    |   |-- Contributing.md
    |   |-- LU_factorisation.cpp
    |   |-- README.md
    |   |-- algoimage.png
    |   |-- broadcast.cpp
    |   |-- concatenate.cpp
    |   |-- input.txt(for max.cpp)
    |   |-- max.cpp
    |   |-- mm.cpp  (Matrix Multiplication implementation)
    |   |-- montecarlo.cpp
    |   |-- mv.cpp (Matrix Vector multiplication)
    |   |-- pi-reduction.cpp (Parallel pi computation using OpenMP reduction)
    |   |-- prod.cpp (Parallel product computation)
    |   |-- standard_dev.cpp
    |   |-- sum2.cpp
    |   |-- sum_task.cpp
    |   |-- vvd.cpp (Vector-Vector Dot product)
    |   |-- wrong_sum.cpp (Demo of a wrong summation example for learning purposes)

### **Advanced**

    .
    |-- include
    |   |-- array_concat.h
    |   |-- dot_product.h
    |   |-- find_max.h
    |   |-- lu_decomposition.h
    |   |-- matrix_mult.h
    |   |-- matrix_vector.h
    |   |-- montecarlo.h
    |   |-- parallel_sum.h
    |   |-- pi_calculator.h
    |   |-- product_log.h
    |   |-- statistics.h
    |   |-- sum2.h
    |   |-- sum_task.h
    |
    |-- src
    |   |-- Histogram/
    |   |   |-- histo.cpp
    |   |   |-- data.txt
    |   |-- Contributing.md
    |   |-- LU_factorisation.cpp
    |   |-- README.md
    |   |-- algoimage.png
    |   |-- broadcast.cpp
    |   |-- concatenate.cpp
    |   |-- input.txt(for max.cpp)
    |   |-- max.cpp
    |   |-- mm.cpp  (Matrix Multiplication implementation)
    |   |-- montecarlo.cpp
    |   |-- mv.cpp (Matrix Vector multiplication)
    |   |-- pi-reduction.cpp (Parallel pi computation using OpenMP reduction)
    |   |-- prod.cpp (Parallel product computation)
    |   |-- standard_dev.cpp
    |   |-- sum2.cpp
    |   |-- sum_task.cpp
    |   |-- vvd.cpp (Vector-Vector Dot product)
    |   |-- wrong_sum.cpp (Demo of a wrong summation example for learning purposes)
    |-- Application
    |   |-- page-rank.cpp
### Note
> Information about Functions in main is provided in [README.md](src/README.md) <br>
> For contributing to this repo kindly go through the guidelines provided in [Contributing.md](Contributing.md)
