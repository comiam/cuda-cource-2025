#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
do {                                                                             \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
        fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__,                \
                cudaGetErrorString(err__));                                      \
        std::abort();                                                            \
    }                                                                            \
} while(0)

#define TILE_SIZE 32

void matmul_cpu(float* A, float* B, float* C, int N, int M, int K) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int l = 0; l < M; l++) {
                sum += A[i * M + l] * B[l * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

__global__ void naive_kernel(float* A, float* B, float* C, 
                                   int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__global__ void shared_kernel(float* A, float* B, float* C,
                                     int N, int M, int K) {
    // shared память
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // цикл по тайлам
    for (int t = 0; t < (M + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // загрузка тайла A
        int a_row = by * TILE_SIZE + ty;
        int a_col = t * TILE_SIZE + tx;
        if (a_row < N && a_col < M) {
            As[ty][tx] = A[a_row * M + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // загрузка тайла B
        int b_row = t * TILE_SIZE + ty;
        int b_col = bx * TILE_SIZE + tx;
        if (b_row < M && b_col < K) {
            Bs[ty][tx] = B[b_row * K + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // умножение тайлов
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < K) {
        C[row * K + col] = sum;
    }
}

void run_naive(float* A, float* B, float* C, int N, int M, int K) {
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * K * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    
    naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);

    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void run_shared(float* A, float* B, float* C, int N, int M, int K) {
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * K * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    
    shared_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);

    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

bool is_equal(float* C1, float* C2, int N, int K) {
    for (int i = 0; i < N * K; i++) {
        if (fabsf(C1[i] - C2[i]) > 1e-4f) {
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    // размеры по умолчанию
    int N = 512, M = 512, K = 512;
    
    if (argc >= 4) {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    float* A = new float[N * M];
    float* B = new float[M * K];
    float* C_naive = new float[N * K];
    float* C_shared = new float[N * K];
    float* C_cpu = new float[N * K];
    
    // инициализация матриц
    for (int i = 0; i < N * M; i++) A[i] = 1.2f;
    for (int i = 0; i < M * K; i++) B[i] = 2.2f;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // замер времени для naive
    CUDA_CHECK(cudaEventRecord(start));
    run_naive(A, B, C_naive, N, M, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    
    // замер времени для shared
    CUDA_CHECK(cudaEventRecord(start));
    run_shared(A, B, C_shared, N, M, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float shared_time;
    CUDA_CHECK(cudaEventElapsedTime(&shared_time, start, stop));

    // замер времени для cpu
    CUDA_CHECK(cudaEventRecord(start));
    matmul_cpu(A, B, C_cpu, N, M, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float cpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&cpu_time, start, stop));
    
    std::cout << "First matrix size: " << N << "x" << M << std::endl;
    std::cout << "Second matrix size: " << M << "x" << K << std::endl;
    std::cout << "Result matrix size: " << N << "x" << K << std::endl << std::endl;
    std::cout << "CPU: " << cpu_time << " ms" << std::endl;
    std::cout << "Naive kernel: " << naive_time << " ms" << std::endl;
    std::cout << "Shared kernel: " << shared_time << " ms" << std::endl << std::endl;
    std::cout << "Naive speedup: x" << cpu_time / naive_time << std::endl;
    std::cout << "Shared speedup: x" << cpu_time / shared_time << std::endl;

    if (is_equal(C_naive, C_cpu, N, K)) {
        std::cout << "Naive kernel is correct" << std::endl;
    } else {
        std::cout << "Naive kernel is incorrect" << std::endl;
    }

    if (is_equal(C_shared, C_cpu, N, K)) {
        std::cout << "Shared kernel is correct" << std::endl;
    } else {
        std::cout << "Shared kernel is incorrect" << std::endl;
    }
    
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_shared;
    delete[] C_cpu;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}