#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <thread>
#include <vector>
#include <mutex>

#define TILE_SIZE 16

std::mutex output_mutex;

void matrixMultiplyCPU(float* A, float* B, float* C, int M, int N, int K) {
    // просто считаем матрицы на ЦП
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

__global__ void matrixMultiplyBase(float* A, float* B, float* C, int M, int N, int K) {
    // просто считаем матрицы на ГПУ
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * K + col];
        }
        
        C[row * K + col] = sum;
    }
}


__global__ void matrixMultiplyTiled(float* A, float* B, float* C, int M, int N, int K) {
    // считаем на ГПУ с тайлингом и shared memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < N) {
            tileA[ty][tx] = A[row * N + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        int bRow = t * TILE_SIZE + ty;
        if (bRow < N && col < K) {
            tileB[ty][tx] = B[bRow * K + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand() % 10);
    }
}

bool verifyResult(float* cpuResult, float* gpuResult, int size, float tolerance = 0.01f) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpuResult[i] - gpuResult[i]) > tolerance) {
            printf("Error at position %d: CPU = %f, GPU = %f\n", i, cpuResult[i], gpuResult[i]);
            return false;
        }
    }
    return true;
}

double getTime() {
    return (double)clock() / CLOCKS_PER_SEC;
}

void runBenchmark(int N, float* h_A, float* h_B, float* h_C_cpu, float* h_C_gpu_basic, 
                  float* h_C_gpu_tiled, float* d_A, float* d_B, float* d_C,
                  cudaEvent_t start, cudaEvent_t stop) {
    size_t bytesA = N * N * sizeof(float);
    size_t bytesB = N * N * sizeof(float);
    size_t bytesC = N * N * sizeof(float);
    
    srand(42);
    initMatrix(h_A, N, N);
    initMatrix(h_B, N, N);
    
    double startCPU = getTime();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N, N, N);
    double endCPU = getTime();
    double timeCPU = endCPU - startCPU;
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    matrixMultiplyBase<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, N, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float timeGpuBasic = 0;
    cudaEventElapsedTime(&timeGpuBasic, start, stop);
    timeGpuBasic /= 1000.0f;
    
    cudaMemcpy(h_C_gpu_basic, d_C, bytesC, cudaMemcpyDeviceToHost);
    bool basicCorrect = verifyResult(h_C_cpu, h_C_gpu_basic, N * N);
    
    cudaEventRecord(start);
    matrixMultiplyTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, N, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float timeGpuTiled = 0;
    cudaEventElapsedTime(&timeGpuTiled, start, stop);
    timeGpuTiled /= 1000.0f;
    
    cudaMemcpy(h_C_gpu_tiled, d_C, bytesC, cudaMemcpyDeviceToHost);
    bool tiledCorrect = verifyResult(h_C_cpu, h_C_gpu_tiled, N * N);
    
    // Синхронизированный вывод
    std::lock_guard<std::mutex> lock(output_mutex);
    printf("\n%dx%d matrices\n", N, N);
    printf("CPU Time: %.4f sec\n", timeCPU);
    printf("GPU basic Time: %.4f sec, result %s\n", timeGpuBasic, basicCorrect ? "correct" : "incorrect!");
    printf("GPU with tiling + shared memory Time: %.4f sec, result %s\n", timeGpuTiled, tiledCorrect ? "correct" : "incorrect!");
    printf("Speedup (basic vs CPU): %.2fx\n", timeCPU / timeGpuBasic);
    printf("Speedup (tiled vs CPU): %.2fx\n", timeCPU / timeGpuTiled);
    printf("Speedup (tiled vs basic): %.2fx\n", timeGpuBasic / timeGpuTiled);
    fflush(stdout);
}

// Обёртка для runBenchmark в потоке
void threadRunBenchmark(int N) {
    int maxN = 4096;
    size_t maxBytes = maxN * maxN * sizeof(float);
    
    float* h_A = (float*)malloc(maxBytes);
    float* h_B = (float*)malloc(maxBytes);
    float* h_C_cpu = (float*)malloc(maxBytes);
    float* h_C_gpu_basic = (float*)malloc(maxBytes);
    float* h_C_gpu_tiled = (float*)malloc(maxBytes);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, maxBytes);
    cudaMalloc(&d_B, maxBytes);
    cudaMalloc(&d_C, maxBytes);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    runBenchmark(N, h_A, h_B, h_C_cpu, h_C_gpu_basic, h_C_gpu_tiled,
                 d_A, d_B, d_C, start, stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_basic);
    free(h_C_gpu_tiled);
}

// Обёртка для прямоугольных матриц в потоке
void threadRunBenchmark2(int M, int N, int K) {
    int maxN = 4096;
    size_t maxBytes = maxN * maxN * sizeof(float);
    
    float* h_A = (float*)malloc(maxBytes);
    float* h_B = (float*)malloc(maxBytes);
    float* h_C_cpu = (float*)malloc(maxBytes);
    float* h_C_gpu_basic = (float*)malloc(maxBytes);
    float* h_C_gpu_tiled = (float*)malloc(maxBytes);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, maxBytes);
    cudaMalloc(&d_B, maxBytes);
    cudaMalloc(&d_C, maxBytes);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    size_t bytesA = M * N * sizeof(float);
    size_t bytesB = N * K * sizeof(float);
    size_t bytesC = M * K * sizeof(float);
    
    srand(42);
    initMatrix(h_A, M, N);
    initMatrix(h_B, N, K);
    
    double startCPU = getTime();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, M, N, K);
    double endCPU = getTime();
    double timeCPU = endCPU - startCPU;
    
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    matrixMultiplyBase<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float timeGpuBasic = 0;
    cudaEventElapsedTime(&timeGpuBasic, start, stop);
    timeGpuBasic /= 1000.0f;
    
    cudaMemcpy(h_C_gpu_basic, d_C, bytesC, cudaMemcpyDeviceToHost);
    bool basicCorrect = verifyResult(h_C_cpu, h_C_gpu_basic, M * K);
    
    cudaEventRecord(start);
    matrixMultiplyTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float timeGpuTiled = 0;
    cudaEventElapsedTime(&timeGpuTiled, start, stop);
    timeGpuTiled /= 1000.0f;
    
    cudaMemcpy(h_C_gpu_tiled, d_C, bytesC, cudaMemcpyDeviceToHost);
    bool tiledCorrect = verifyResult(h_C_cpu, h_C_gpu_tiled, M * K);
    
    // Синхронизированный вывод
    std::lock_guard<std::mutex> lock(output_mutex);
    printf("\n%dx%d × %dx%d matrices\n", M, N, N, K);
    printf("CPU Time: %.4f sec\n", timeCPU);
    printf("GPU basic Time: %.4f sec, result %s\n", timeGpuBasic, basicCorrect ? "correct" : "incorrect!");
    printf("GPU with tiling + shared memory Time: %.4f sec, result %s\n", timeGpuTiled, tiledCorrect ? "correct" : "incorrect!");
    printf("Speedup (basic vs CPU): %.2fx\n", timeCPU / timeGpuBasic);
    printf("Speedup (tiled vs CPU): %.2fx\n", timeCPU / timeGpuTiled);
    printf("Speedup (tiled vs basic): %.2fx\n", timeGpuBasic / timeGpuTiled);
    fflush(stdout);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_basic);
    free(h_C_gpu_tiled);
}

int main() {
    fflush(stdout);
    
    std::vector<std::thread> threads;
    
    // Запускаем каждый бенчмарк в отдельном потоке
    threads.push_back(std::thread(threadRunBenchmark, 32));
    threads.push_back(std::thread(threadRunBenchmark, 128));
    threads.push_back(std::thread(threadRunBenchmark, 512));
    threads.push_back(std::thread(threadRunBenchmark, 1024));
    threads.push_back(std::thread(threadRunBenchmark, 2048));
    threads.push_back(std::thread(threadRunBenchmark, 4096));
    threads.push_back(std::thread(threadRunBenchmark2, 128, 256, 128));
    
    // Ждём завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }

    fflush(stdout);
    return 0;
}