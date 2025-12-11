#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)


void fillMatrix(float* mat, int rows, int cols);
bool verifyResult(const float* C_cpu, const float* C_gpu, int M, int N, float tolerance = 1e-3f);
void matMulCPU(const float* A, const float* B, float* C, int M, int K, int N);
__global__ void matMulKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N);


int main(int argc, char* argv[]) {
    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    if (M <= 0 || K <= 0 || N <= 0) {
        std::cerr << "Ошибка: все размеры должны быть положительными целыми числами." << std::endl;
        return 1;
    }

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_cpu = (float*)malloc(size_C);
    float *h_C_gpu = (float*)malloc(size_C);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        std::cerr << "Ошибка выделения памяти на CPU!" << std::endl;
        return -1;
    }

    srand(time(nullptr));
    fillMatrix(h_A, M, K);
    fillMatrix(h_B, K, N);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    matMulCPU(h_A, h_B, h_C_cpu, M, K, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    CUDA_CHECK(cudaEventRecord(start_gpu));
    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

    float gpu_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu));

    std::cout << "Матрицы: A[" << M << "x" << K << "] * B[" << K << "x" << N
              << "] = C[" << M << "x" << N << "]" << std::endl;
    std::cout << "Время на CPU: " << cpu_time_ms / 1000.0f << " сек" << std::endl;
    std::cout << "Время на GPU: " << gpu_time_ms / 1000.0f << " сек" << std::endl;
    if (gpu_time_ms > 0)
        std::cout << "Ускорение: " << (cpu_time_ms / gpu_time_ms) << "x" << std::endl;

    if (verifyResult(h_C_cpu, h_C_gpu, M, N)) {
        std::cout << "Результат корректен!" << std::endl;
    } else {
        std::cout << "Ошибка в вычислениях!" << std::endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaEventDestroy(start_gpu); cudaEventDestroy(stop_gpu);

    return 0;
}
