#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <chrono>


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
#define TILE 16
#endif

// Оптимизированная версия перемножения матриц
__global__ void gpu_optimised_mm(const int *A, const int *B, int *C, int m, int n, int k)
{
    // shared tiles (+1 to mitigate bank conflicts)
    __shared__ int tile_a[TILE][TILE + 1];
    __shared__ int tile_b[TILE][TILE + 1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE + ty;   // Глобальный индекс строки для матрицы C
    int col = bx * TILE + tx;   // Глобальный индекс колонки для матрицы C

    long long acc = 0;

    // Количество Тайлов для покрытия "среднего" измерения (n)
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        // Глобальный индекс кколонки дла А и глобальный индекс колонки для B
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        // Загружаем элемент Тайла из A: A[row, a_col]
        if (row < m && a_col < n)
            tile_a[ty][tx] = A[row * n + a_col];
        else
            tile_a[ty][tx] = 0;

        // Загружаем элемент Тайла из B: B[b_row, col]
        if (b_row < n && col < k)
            tile_b[ty][tx] = B[b_row * k + col];
        else
            tile_b[ty][tx] = 0;

        __syncthreads();

        // Перемножаем два виктора шириной с Тайл
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

void fill_matrix_manual(int* M, int rows, int cols) {
    printf("Заполнение матрицы вручную (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("[%d][%d] = ", i, j);
            std::cin >> M[i * cols + j];
        }
    }
}


void print_matrix(const int* M, int rows, int cols, const char* name) {
    printf("Matrix %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%5d ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main() {
    int m = 48;
    int n = 32;
    int k = 48;

    srand(3333);

    int *h_a, *h_b, *h_c_naive, *h_c_opt, *h_cc;
    cudaMallocHost(&h_a, sizeof(int)*m*n);
    cudaMallocHost(&h_b, sizeof(int)*n*k);
    cudaMallocHost(&h_c_naive, sizeof(int)*m*k);
    cudaMallocHost(&h_c_opt, sizeof(int)*m*k);
    cudaMallocHost(&h_cc, sizeof(int)*m*k);

    // Рандомно заполняем A и B
    for (int i = 0; i < m*n; i++) h_a[i] = rand() % 1024;
    for (int i = 0; i < n*k; i++) h_b[i] = rand() % 1024;

	// Ручное заполнение матриц
	// fill_matrix_manual(h_a, m, n);
	// fill_matrix_manual(h_b, n, k);

    // Device memory
    int *d_a, *d_b, *d_c_naive, *d_c_opt;
    cudaMalloc(&d_a, sizeof(int)*m*n);
    cudaMalloc(&d_b, sizeof(int)*n*k);
    cudaMalloc(&d_c_naive, sizeof(int)*m*k);
    cudaMalloc(&d_c_opt, sizeof(int)*m*k);

    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);


    unsigned int grid_rows = (m + TILE - 1) / TILE;
    unsigned int grid_cols = (k + TILE - 1) / TILE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(TILE, TILE);

    // Запускаем тесты параллельно
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t n_start, n_stop, o_start, o_stop;
    cudaEventCreate(&n_start);
    cudaEventCreate(&n_stop);
    cudaEventCreate(&o_start);
    cudaEventCreate(&o_stop);

    // Наивное GPU умножение
    cudaEventRecord(n_start, stream1);
    gpu_naive_mm<<<dimGrid, dimBlock, 0, stream1>>>(
        d_a, d_b, d_c_naive, m, n, k
    );
    cudaEventRecord(n_stop, stream1);

    // Оптимизированное GPU умножение
    cudaEventRecord(o_start, stream2);
    gpu_optimised_mm<<<dimGrid, dimBlock, 0, stream2>>>(
        d_a, d_b, d_c_opt, m, n, k
    );
    cudaEventRecord(o_stop, stream2);


    CUDA_CHECK(cudaEventSynchronize(n_stop));
    CUDA_CHECK(cudaEventSynchronize(o_stop));


    cudaMemcpyAsync(h_c_naive, d_c_naive, sizeof(int)*m*k, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_c_opt,   d_c_opt,   sizeof(int)*m*k, cudaMemcpyDeviceToHost, stream2);

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    float naive_ms = 0, opt_ms = 0;
    cudaEventElapsedTime(&naive_ms, n_start, n_stop);
    cudaEventElapsedTime(&opt_ms, o_start, o_stop);

    printf("GPU naive      time: %.3f ms\n", naive_ms);
    printf("GPU optimized  time: %.3f ms\n", opt_ms);

    // CPU замер

    auto start = std::chrono::high_resolution_clock::now();
    cpu_matmul(h_a, h_b, h_cc, m, n, k);
	auto stop = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> ms = stop - start;
    printf("CPU time: %.3f ms\n", ms.count());

    // Проверка CPU перемножения с оптимизированным и наивными перемножениями
    int ok = 1;
    for (int i = 0; i < m*k; ++i) {
        if (h_cc[i] != h_c_opt[i]) {
            ok = 0;
            break;
        }
    }

	for (int i = 0; i < m*k; ++i) {
        if (h_cc[i] != h_c_naive[i]) {
            ok = 0;
            break;
        }
    }

    if (ok)
        printf("Results are correct. GPU speedup = %.3f\n", naive_ms / opt_ms);
    else
        printf("Incorrect results!\n");

	//print_matrix(h_a, m, n, "A");
    //print_matrix(h_b, n, k, "B");
    //print_matrix(h_cc, m, k, "C");

    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_naive);
    cudaFree(d_c_opt);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c_naive);
    cudaFreeHost(h_c_opt);
    cudaFreeHost(h_cc);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaEventDestroy(n_start);
    cudaEventDestroy(n_stop);
    cudaEventDestroy(o_start);
    cudaEventDestroy(o_stop);

    return 0;
}
