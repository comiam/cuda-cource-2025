#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#pragma once


#define BLOCK_SIZE 16

#define CUDA_CHECK(call)                                                         \
do {                                                                             \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
        fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__,                \
                cudaGetErrorString(err__));                                      \
        std::abort();                                                            \
    }                                                                            \
} while(0)


void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << name << ":\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << std::setw(10) << matrix[row * cols + col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::defaultfloat;
}

__global__ void matrix_multiplication_kernel(const float* A,
                                             const float* B,
                                             float* C,
                                             int M,
                                             int N,
                                             int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index
    
  

    __shared__ float sm_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sm_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int tiled_col = tile_idx * BLOCK_SIZE + threadIdx.x;
        if (y < M && tiled_col < K) {
            sm_A[threadIdx.y][threadIdx.x] = A[y * K + tiled_col];
        } else {
            sm_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int tiled_row = tile_idx * BLOCK_SIZE + threadIdx.y;
        if (tiled_row < K && x < N) {
            sm_B[threadIdx.y][threadIdx.x] = B[tiled_row * N + x];
        } else {
            sm_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += sm_A[threadIdx.y][i] * sm_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (x < N && y < M) {
        C[y * N + x] = sum;
    }
}

int main() 
    {
        int M = 0;
        int K = 0;
        int N = 0;
        std::cout << "Enter matrix sizes M K N (A: MxK, B: KxN): ";
        if (!(std::cin >> M >> K >> N) || M <= 0 || K <= 0 || N <= 0) {
            std::cerr << "Invalid sizes.\n";
            return EXIT_FAILURE;
            
        }

        const int size_A = M * K;
        const int size_B = K * N;
        const int size_C = M * N;

        float* host_A = new float[size_A];
        float* host_B = new float[size_B];
        float* host_C = new float[size_C];

        for(int i = 0; i < size_A; ++i) {
            host_A[i] =2;
        }
        for(int i = 0; i < size_B; ++i) {
            host_B[i] =2;
        }

        float* dev_A = nullptr;
        float* dev_B = nullptr;
        float* dev_C = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_A, size_A * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_B, size_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_C, size_C * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dev_A, host_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_B, host_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        matrix_multiplication_kernel<<<grid, block>>>(dev_A, dev_B, dev_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(host_C, dev_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "Matrix multiplication completed successfully on GPU." << std::endl;

        print_matrix(host_A, M, K, "Matrix A");
        print_matrix(host_B, K, N, "Matrix B");
        print_matrix(host_C, M, N, "Matrix C");

        CUDA_CHECK(cudaFree(dev_A));
        CUDA_CHECK(cudaFree(dev_B));
        CUDA_CHECK(cudaFree(dev_C));
        delete[] host_A;
        delete[] host_B;
        delete[] host_C;

        return 0;
   
}

