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


//Базовая версия на CPU
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // Создаём C
    for (int i = 0; i < M*K; ++i) C[i] = 0.0f;

    for (int i = 0; i < M; ++i) {
        for (int p = 0; p < N; ++p) {
            float aip = A[i*N + p];
            // перемножаем строку i матрицы A со строкой p матрицы B (B строка p относитя к колонкам)
            const float* Bp = B + p*K;
            float* Ci = C + i*K;
            for (int j = 0; j < K; ++j) {
                Ci[j] += aip * Bp[j];
            }
        }
    }
}



//Базовая версия перемножения на GPU без оптимизации
// &a - GPU указатель на m*n матрицу A
// $b - GPU указатель на n*k матрицу B
// &c - GPU выходной указатель на результирующую m*k матрицу C
__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
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
