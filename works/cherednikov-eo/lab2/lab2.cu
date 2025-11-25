#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "error_check.cuh"

// &A - указатель на m*n матрицу A
// &B - указатель на n*k матрицу B
// &C - выходной указатель на результирующую m*k матрицу C

//Базовая версия на CPU
void cpu_matmul(int* A, int* B, int* C, int m, int n, int m) {
    // Создаём C
    for (int i = 0; i < m*k; ++i) C[i] = 0;

    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < n; ++p) {
            int aip = A[i*n + p];
            // перемножаем строку i матрицы A со строкой p матрицы B (B строка p относитя к колонкам)
            int* Bp = B + p*k;
            int* Ci = C + i*k;
            for (int j = 0; j < K; ++j) {
                Ci[j] += aip * Bp[j];
            }
        }
    }
}


//Базовая версия перемножения на GPU без оптимизации
__global__ void gpu_naive_mm(int *A,int *B, int *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M || col >= K) return;

    int sum = 0;
    for(int i = 0; i < n; i++) {
    	sum += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = sum;
}


#ifndef TILE
#define TILE 32
#endif

// Оптимизированная версия перемножения матриц
__global__ void gpu_optimised_mm(int *A,int *B, int *C, int m, int n, int k)
{
    __shared__ int sA[TILE][TILE+1];
    __shared__ int sB[TILE][TILE+1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int sum = 0;

	//Tile A: rows: [blockY*TILE .. blockY*TILE+TILE-1], cols: [m*TILE .. m*x]

	//Проходимся по всем тайлам
    for(int i = 0; i < gridDim.x; ++i) {

		idx = row * n + i * TILE + threadIdx.x;

		if (idx )

    	sum += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = sum;
}


int main() {
    int m, n, k;

    size_t size = width * height;
    char* h_canvas = (char*)malloc(size);
    char* d_canvas;

    cudaMalloc(&d_canvas, size);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    draw_square<<<grid, block>>>(d_canvas, width, height, margin);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
