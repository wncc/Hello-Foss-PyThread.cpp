#include <iostream>
#include <omp.h>
#include <vector>
#include <stdexcept>

// Function to broadcast two matrices
void broadcast(const std::vector<std::vector<int>>& A, std::vector<std::vector<int>>& B) {
    if (A.empty() || B.empty()) {
        throw std::invalid_argument("Matrix A or B cannot be empty");
    }

    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t rowsB = B.size();
    size_t colsB = B[0].size();

    if (rowsA != rowsB && rowsB != 1) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting (rows)");
    }
    if (colsA != colsB && colsB != 1) {
        throw std::invalid_argument("Incompatible dimensions for broadcasting (columns)");
    }

    if (rowsB == 1) {
        B.resize(rowsA, B[0]);
    }
    if (colsB == 1) {
        for (auto& row : B) {
            row.resize(colsA, row[0]);
        }
    }

   
    #pragma omp parallel for
    for (size_t i = 0; i < rowsA; i++) {
        #pragma omp parallel for
        for (size_t j = 0; j < colsA; j++) {
            B[i][j] = A[i][j]+B[i][j];
        }
    }
}

int main() {
    std::vector<std::vector<int>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<int>> B = {{1,2,3}}; // B has only one row

    try {
        broadcast(A, B);
        for (const auto& row : B) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
