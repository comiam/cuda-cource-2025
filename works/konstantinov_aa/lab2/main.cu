#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono> // Added for CPU timing
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


// Функция для вывода матрицы в консоль
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

// Реализация перемножения матриц на CPU для проверки корректности
void matrix_mult_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA Kernel для перемножения матриц с использованием Shared Memory
// Использует технику тайлинга (tiling) для оптимизации доступа к памяти
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Индекс столбца
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Индекс строки
    
  
    // Выделение разделяемой памяти для тайлов матриц A и B
    __shared__ float sm_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sm_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    // Вычисляем количество тайлов, необходимых для прохода по общему измерению K
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Загрузка данных в Shared Memory
        
        // Загрузка элемента из матрицы A
        int tiled_col = tile_idx * BLOCK_SIZE + threadIdx.x;
        if (y < M && tiled_col < K) {
            sm_A[threadIdx.y][threadIdx.x] = A[y * K + tiled_col];
        } else {
            sm_A[threadIdx.y][threadIdx.x] = 0.0f; // Заполнение нулями за пределами матрицы
        }

        // Загрузка элемента из матрицы B
        int tiled_row = tile_idx * BLOCK_SIZE + threadIdx.y;
        if (tiled_row < K && x < N) {
            sm_B[threadIdx.y][threadIdx.x] = B[tiled_row * N + x];
        } else {
            sm_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Барьер синхронизации: ждем, пока все потоки загрузят свои данные в Shared Memory
        __syncthreads();

        // Вычисление частичной суммы для текущего тайла
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += sm_A[threadIdx.y][i] * sm_B[i][threadIdx.x];
        }
        
        // Барьер синхронизации перед загрузкой следующего тайла
        __syncthreads();
    }

    // Запись результата в глобальную память
    if (x < N && y < M) {
        C[y * N + x] = sum;
    }
}

int main() {
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

    // Инициализация матриц случайными значениями
    srand(time(NULL));
    for(int i = 0; i < size_A; ++i) {
        host_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for(int i = 0; i < size_B; ++i) {
        host_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* dev_A = nullptr;
    float* dev_B = nullptr;
    float* dev_C = nullptr;
    // Выделение памяти на GPU
    CUDA_CHECK(cudaMalloc(&dev_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_C, size_C * sizeof(float)));

    // Копирование данных с Host на Device
    CUDA_CHECK(cudaMemcpy(dev_A, host_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, host_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

    // Настройка сетки и блоков
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Создание событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_multiplication_kernel<<<grid, block>>>(dev_A, dev_B, dev_C, M, N, K);
    cudaEventRecord(stop);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Проверка ошибок запуска ядра

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms\n";

    // Копирование результата обратно на Host
    CUDA_CHECK(cudaMemcpy(host_C, dev_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // Проверка на CPU для матриц небольшого размера
    if (M <= 512 && N <= 512 && K <= 512) {
        float* cpu_C = new float[size_C];
        auto start_cpu = std::chrono::high_resolution_clock::now();
        matrix_mult_cpu(host_A, host_B, cpu_C, M, N, K);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;
        std::cout << "CPU Time: " << cpu_ms.count() << " ms\n";
        std::cout << "Speedup: " << cpu_ms.count() / milliseconds << "x\n";

        float max_error = 0.0f;
        for(int i=0; i<size_C; ++i) max_error = std::max(max_error, std::abs(host_C[i] - cpu_C[i]));
        std::cout << "Max Absolute Error: " << max_error << "\n";
        delete[] cpu_C;
    }

    // Вывод матриц только если они очень маленькие
    if (M <= 10 && N <= 10) {
        print_matrix(host_A, M, K, "Matrix A");
        print_matrix(host_B, K, N, "Matrix B");
        print_matrix(host_C, M, N, "Matrix C");
    }

    // Освобождение памяти
    CUDA_CHECK(cudaFree(dev_A));
    CUDA_CHECK(cudaFree(dev_B));
    CUDA_CHECK(cudaFree(dev_C));
    delete[] host_A;
    delete[] host_B;
    delete[] host_C;

    return 0;
}
