#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>
#include <mpi.h>
#include <iostream>
#include<algorithm>
using namespace std;

int NumberOfPoints = 0;
int max_value = -1;
int *p;
int *arr;

int* ReadFromFile()
{
    int *points;
    FILE *file;
    int index = 0;
    char line[10];
    file = fopen("data.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }
    while (fgets(line, sizeof(line), file))
        NumberOfPoints++;
    fclose(file);
    
    points = (int *)malloc(NumberOfPoints * sizeof(int)); // Allocate memory for points
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
    
    while (fgets(line, sizeof(line), file)) {
        points[index++] = atoi(line);
    }
    fclose(file);
    return points;
}

int main(int argc, char **argv)
{
    int indexx = 0;
    int *points, Bars, np, Range, tmp_Range = 0,
        Points_per_process;
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

    if (rank == 0)
    {
        cout << "Enter the number of bars" << endl;
        cin >> Bars;

        points = ReadFromFile();
        Points_per_process = ((double)NumberOfPoints / (NumberOfprocess)) + 0.5;

        size = Points_per_process * NumberOfprocess;
        if (size < NumberOfPoints)
            size = NumberOfPoints;

        p = (int*)malloc(size * sizeof(int)); 
        for (i = 0; i < size; i++)
        {
            if (i < NumberOfPoints)
                p[i] = points[i];
            else
                p[i] = -1;
        }
        
        max_value = *std::max_element(p, p + NumberOfPoints);  // Find max value for Range
        Range = max_value / Bars;
        if (max_value % Bars != 0)
        {
            Range++;
        }
        free(points);
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Bars, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Range, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Points_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

    irecv = (int*)malloc(Points_per_process * sizeof(int)); 
    count = (int*)malloc(Bars * sizeof(int)); 
    for (l = 0; l < Bars; l++)
    {
        count[l] = 0;
    }

    MPI_Scatter(p, Points_per_process, MPI_INT, irecv, Points_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    // Dynamically allocate arr based on Points_per_process
    arr = (int*)malloc(Points_per_process * sizeof(int));

#pragma omp parallel shared(count)
    {
#pragma omp for schedule(static)
        for (i = 0; i < Points_per_process; i++)
        {
            for (l = 0; l < Bars; l++)
            {
                if (irecv[i] <= (l + 1) * Range && irecv[i] != -1)
                {
                    count[l]++;
                    arr[i] = l;  // Store the bar index in arr
                    break;
                }
            }
        }
    }

    AllCount = (int*)malloc(NumberOfPoints * sizeof(int));  // Gather result

    MPI_Gather(arr, Points_per_process, MPI_INT, AllCount, Points_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Count the final distribution of points across bars
        for (i = 0; i < Bars; i++)
        {
            count[i] = 0;
        }

        for (i = 0; i < NumberOfPoints; i++)
        {
            if (AllCount[i] != -1)
            {
                count[AllCount[i]]++;
            }
        }

        for (i = 0; i < Bars; i++)
        {
            cout << "Bar " << i << " has " << count[i] << " points" << endl;
        }

        free(AllCount);
    }

    free(irecv);
    free(count);
    free(arr);

    MPI_Finalize();
    return 0;
}
