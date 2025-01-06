#include "find_max.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <vector>
#include <cstdlib>

// Function to find the maximum in a subset of the array
int findMax(int elementsToProcess, const std::vector<int>& data) {
    int max = data[0];
    for (int i = 0; i < elementsToProcess; i++) {
        if (data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  
    int processorsNr;
    MPI_Comm_size(MPI_COMM_WORLD, &processorsNr);  
    int processId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processId); 

    int buff;  
    double time;

    if (processId == 0) {
        int max = 0;  // Global maximum
        std::ifstream inputs("input.txt"); 
        int totalElements = 0;
        int elementsToEachProcess = 0;

        if (!inputs.is_open()) {
            std::cerr << "Error: Input file not found" << std::endl;
            MPI_Finalize();
            return 0;
        }

        std::vector<int> data; 
        int num;
        while (inputs >> num) {
            data.push_back(num);  
            totalElements++;
        }
        inputs.close();

        if (totalElements == 0) {
            std::cerr << "Error: No data in input file" << std::endl;
            MPI_Finalize();
            return 0;
        }

        elementsToEachProcess = totalElements / processorsNr;
        int remainder = totalElements % processorsNr;  // Remaining elements if not divisible

        // Send data to worker processes
        for (int i = 1; i < processorsNr; i++) {
            int startIdx = i * elementsToEachProcess;
            int count = (i == processorsNr - 1) ? elementsToEachProcess + remainder : elementsToEachProcess;
            buff = count;

            MPI_Send(&buff, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(data.data() + startIdx, buff, MPI_INT, i, 2, MPI_COMM_WORLD);
        }

        // Synchronize all processes before starting the timing
        MPI_Barrier(MPI_COMM_WORLD);
        time = MPI_Wtime();  // Start timing

        // Master process finds its own max
        max = findMax(elementsToEachProcess + remainder, data);

        // Receive max values from worker processes
        int theMax;
        for (int i = 1; i < processorsNr; i++) {
            MPI_Recv(&theMax, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (theMax > max) {
                max = theMax;
            }
        }

        time = MPI_Wtime() - time;  // Stop timing
        std::cout << "Global maximum is " << max << std::endl;
        std::cout << "Time elapsed: " << time << " seconds" << std::endl;
    }
    
    else {
        // Worker processes: receive data and find max
        MPI_Recv(&buff, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<int> dataToProcess(buff);  
        MPI_Recv(dataToProcess.data(), buff, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Synchronize before starting the computation
        MPI_Barrier(MPI_COMM_WORLD);

        int theMax = findMax(buff, dataToProcess);

        // Send local max to master process
        MPI_Send(&theMax, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
