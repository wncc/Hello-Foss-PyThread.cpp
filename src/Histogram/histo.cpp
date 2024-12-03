#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
using namespace std;

int NumberOfPoints = 0;
int max_value = -1;
int *p;
int arr[100];

int *ReadFromFile() {
    int *points;
    FILE *file;
    int index = 0;
    char line[10];

    file = fopen("data.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Count the number of points
    while (fgets(line, sizeof(line), file)) {
        NumberOfPoints++;
    }
    fclose(file);

    points = (int *)malloc(NumberOfPoints * sizeof(int));
    if (points == NULL) {
        perror("Error allocating memory");
        return NULL;
    }

    file = fopen("data.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        free(points);
        return NULL;
    }

    // Read points into the array
    while (fgets(line, sizeof(line), file)) {
        points[index++] = atoi(line);
        if (points[index - 1] > max_value) {
            max_value = points[index - 1];
        }
    }
    fclose(file);
    return points;
}

int main(int argc, char **argv) {
    int indexx = 0;
    int *points, Bars, np, Range, Points_per_process;
    int size;
    int *irecv;
    int *AllCount, *count;
    int rank, NumberOfprocess, i, l, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NumberOfprocess);

#pragma omp parallel
    {
        np = omp_get_num_threads();
    }

    if (rank == 0) {
        cout << "Enter the number of bars: ";
        cin >> Bars;

        points = ReadFromFile();
        if (points == NULL) {
            MPI_Finalize();
            return -1;
        }

        Points_per_process = (NumberOfPoints + NumberOfprocess - 1) / NumberOfprocess; // Ceiling division
        size = Points_per_process * NumberOfprocess;
        p = (int *)malloc(size * sizeof(int));

        for (i = 0; i < size; i++) {
            if (i < NumberOfPoints) {
                p[i] = points[i];
            } else {
                p[i] = -1; // Padding with -1
            }
        }

        Range = (max_value + Bars - 1) / Bars; // Ceiling division
        free(points);
    }

    // Broadcast values to all processes
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Bars, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Range, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Points_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local points and counts
    irecv = (int *)malloc(Points_per_process * sizeof(int));
    count = (int *)malloc(Bars * sizeof(int));
    for (l = 0; l < Bars; l++) {
        count[l] = 0; // Initialize counts
    }

    // Scatter data among processes
    MPI_Scatter(p, Points_per_process, MPI_INT, irecv, Points_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    // Histogram computation using OpenMP
#pragma omp parallel for schedule(static) reduction(+ : indexx)
    for (i = 0; i < Points_per_process; i++) {
        for (l = 0; l < Bars; l++) {
            if (irecv[i] != -1 && irecv[i] <= (l + 1) * Range) {
#pragma omp atomic
                count[l]++;
                break;
            }
        }
    }

    free(irecv);

    // Gather results at the root process
    AllCount = (rank == 0) ? (int *)malloc(Bars * NumberOfprocess * sizeof(int)) : NULL;
    MPI_Gather(count, Bars, MPI_INT, AllCount, Bars, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int *finalCounts = (int *)malloc(Bars * sizeof(int));
        for (l = 0; l < Bars; l++) {
            finalCounts[l] = 0;
            for (i = 0; i < NumberOfprocess; i++) {
                finalCounts[l] += AllCount[i * Bars + l];
            }
        }

        // Display the histogram
        for (i = 0; i < Bars; i++) {
            cout << "Bar " << i << " (" << i * Range << " - " << (i + 1) * Range << ") has " << finalCounts[i] << " points." << endl;
        }

        free(finalCounts);
        free(AllCount);
        free(p);
    }

    free(count);
    MPI_Finalize();
    return 0;
}
