#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#include "utils.h"


using namespace std;


__global__ 
void matrixMultiplication(float* A, float* B, float* C, int matrix_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < matrix_size && col < matrix_size) {
        float sum = 0.0f;
        for(int k = 0; k < matrix_size; k++) {
            sum += A[row * matrix_size + k] * B[k * matrix_size + col];
        }
        C[row * matrix_size + col] = sum;
    }
}


double multiplyMatrices(
    vector<vector<float>> &A, vector<vector<float>> &B,vector<vector<float>> &C, int blockDim
){
    int matrix_size = A.size();

    float *A_device, *B_device, *res_device;
    cudaMalloc(&A_device, matrix_size * matrix_size * sizeof(float));
    cudaMalloc(&B_device, matrix_size * matrix_size * sizeof(float));
    cudaMalloc(&res_device, matrix_size * matrix_size * sizeof(float));

    // Копируем матрицы на устройство
    cudaMemcpy(A_device, &A[0][0], matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, &B[0][0], matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    // Определяем размер блока и сетки
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((matrix_size + blockSize.x - 1) / blockSize.x, (matrix_size + blockSize.y - 1) / blockSize.y);

    // Запускаем ядро
    auto start_time = chrono::high_resolution_clock::now();
    matrixMultiplication<<<gridSize, blockSize>>>(A_device, B_device, res_device, matrix_size);
    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;

    cudaDeviceSynchronize();
    cudaMemcpy(&C[0][0], res_device, matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождаем память устройства
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(res_device);

    return elapsed.count()
}

int main(){

    init_random_generator();

    int MATRIX_SIZE = 16;
    auto A = generateMatrix<double>(MATRIX_SIZE);
    auto B = generateMatrix<double>(MATRIX_SIZE);

    vector<vector<double>> res(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
    double elapsed = multiplyMatrices<double>(A, B, res, 4);
    cout << "Простое умножение заняло: " << elapsed << " секунд.\n";


    return 0;
}