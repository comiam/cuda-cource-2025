#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>


#include <cuda_runtime.h>
#include "error_check.cuh"

// &A - указатель на m*n матрицу A
// &B - указатель на n*k матрицу B
// &C - выходной указатель на результирующую m*k матрицу C

//Базовая версия на CPU
void cpu_matmul(int* A, int* B, int* C, int m, int n, int k) {
    // Создаём C
    for (int i = 0; i < m*k; ++i) C[i] = 0;

    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < n; ++p) {
            int aip = A[i*n + p];
            // перемножаем строку i матрицы A со строкой p матрицы B (B строка p относитя к колонкам)
            int* Bp = B + p*k;
            int* Ci = C + i*k;
            for (int j = 0; j < k; ++j) {
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
	if (row >= m || col >= k) return;

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
__global__ void gpu_optimised_mm(const int *A, const int *B, int *C, int m, int n, int k)
{
    // shared tiles (optional +1 to mitigate bank conflicts)
    __shared__ int tile_a[TILE][TILE];
    __shared__ int tile_b[TILE][TILE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE + ty;   // global row index in C
    int col = bx * TILE + tx;   // global col index in C

    int acc = 0;

    // Number of tiles to cover the 'n' dimension (A columns / B rows)
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        // Global column index for A to load and global row index for B to load
        int a_col = t * TILE + tx;       // column in A (and index along n)
        int b_row = t * TILE + ty;       // row in B

        // Load tile element from A: A[row, a_col]
        if (row < m && a_col < n)
            tile_a[ty][tx] = A[row * n + a_col];
        else
            tile_a[ty][tx] = 0;

        // Load tile element from B: B[b_row, col]
        if (b_row < n && col < k)
            tile_b[ty][tx] = B[b_row * k + col];
        else
            tile_b[ty][tx] = 0;

        __syncthreads();

        // Multiply the two TILE-wide vectors
        #pragma unroll
        for (int j = 0; j < TILE; ++j) {
            acc += tile_a[ty][j] * tile_b[j][tx];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = acc;
    }
}


int main() {
    int m = 32;
	int n = 48;
    int k = 32;

	srand(3333);

	// Аллоцируем память для матриц на Хосте и для результат вычислений на ЦПУ
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

	// random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

	// random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

	float gpu_elapsed_time_ms, cpu_elapsed_time_ms, gpu_opt_time_ms;

    // Заводим ивенты для расчётов по времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Аллоцируем память на device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);

    // Передаём матрицы на device
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + TILE - 1) / TILE;
    unsigned int grid_cols = (k + TILE - 1) / TILE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(TILE, TILE);

	cudaEventRecord(start, 0);
	gpu_naive_mm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

	cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaThreadSynchronize());

	//считаем время
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);

	//Считаем для CPU

	cudaEventRecord(start, 0);

    cpu_matmul(h_a, h_b, h_cc, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

	//Пробуем оптимизированную версию
	cudaEventRecord(start, 0);

	gpu_optimised_mm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

	cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaThreadSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_opt_time_ms, start, stop);
	printf("Time elapsed on OPTIMIZED GPU matrix multiplication (%dx%d . %dx%d): %f ms.\n\n",
       	m, n, n, k, gpu_opt_time_ms);


	int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }

	cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
