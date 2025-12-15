#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matrixMulNaive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMulShared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    for (int p = 0; p < (K + TILE_SIZE - 1) / TILE_SIZE; ++p) {
        if (row < M && (p * TILE_SIZE + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + p * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (p * TILE_SIZE + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B[(p * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
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

int main() {
    int M = 2048;
    int N = 512;
    int K = 1024;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_CPU = (float*)malloc(sizeC);
    float* h_C_GPU = (float*)malloc(sizeC);

    srand(2024);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    std::cout << "Matrix size: " << M << " x " << N << std::endl;
    std::cout << "Calculating on CPU..." << std::endl;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_CPU, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
    
    std::cout << "CPU Time: " << std::fixed << std::setprecision(4) << cpu_time << " s" << std::endl;

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time_naive = 0, gpu_time_shared = 0;

    cudaEventRecord(start);
    matrixMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_naive, start, stop);
    gpu_time_naive /= 1000.0f;

    std::cout << "GPU Naive Time: " << gpu_time_naive << " s" << std::endl;
    cudaMemset(d_C, 0, sizeC);

    cudaEventRecord(start);
    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_shared, start, stop);
    gpu_time_shared /= 1000.0f;

    std::cout << "GPU Shared Memory Time: " << gpu_time_shared << " s" << std::endl;

    cudaMemcpy(h_C_GPU, d_C, sizeC, cudaMemcpyDeviceToHost);

    double max_diff = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(h_C_CPU[i] - h_C_GPU[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Correctness check max Error: " << max_diff << std::endl;

    std::cout << "Speedup GPU Naive vs CPU: " << cpu_time / gpu_time_naive << "x" << std::endl;
    std::cout << "Speedup GPU Shared vs CPU: " << cpu_time / gpu_time_shared << "x" << std::endl;
    std::cout << "Speedup Shared vs Naive: " << gpu_time_naive / gpu_time_shared << "x" << std::endl;

    free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
