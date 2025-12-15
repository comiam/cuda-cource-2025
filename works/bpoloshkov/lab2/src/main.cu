#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define BLOCK_SIZE 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void matmulCPU(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matmulGPU(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmulShared(float* A, float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        int bRow = t * BLOCK_SIZE + threadIdx.y;

        if (row < N && aCol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrix(float* M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = (float)(rand() % 100) / 100.0f;
    }
}

bool verifyResult(float* C1, float* C2, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabsf(C1[i] - C2[i]) > 1e-3f) {
            return false;
        }
    }
    return true;
}

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    int N = 1024;

    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 32 || N > 4096) {
            fprintf(stderr, "Error: N must be between 32 and 4096\n");
            return 1;
        }
    }

    size_t size = N * N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);
    float* h_C_shared = (float*)malloc(size);

    srand(42);
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    printf("Matrix size: %dx%d\n\n", N, N);

    double startCPU = getTime();
    if (N <= 512) {
        matmulCPU(h_A, h_B, h_C_cpu, N);
    }
    double timeCPU = getTime() - startCPU;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeGPU;
    cudaEventElapsedTime(&timeGPU, start, stop);
    timeGPU /= 1000.0f;

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    cudaEventRecord(start);
    matmulShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeShared;
    cudaEventElapsedTime(&timeShared, start, stop);
    timeShared /= 1000.0f;

    CUDA_CHECK(cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost));

    if (N <= 512) {
        printf("CPU time:              %.4f sec\n", timeCPU);
    } else {
        printf("CPU time:              skipped (N > 512)\n");
    }
    printf("GPU time (naive):      %.4f sec\n", timeGPU);
    printf("GPU time (shared mem): %.4f sec\n", timeShared);

    if (N <= 512) {
        bool correct = verifyResult(h_C_cpu, h_C_gpu, N);
        printf("\nVerification (naive):  %s\n", correct ? "PASSED" : "FAILED");
        correct = verifyResult(h_C_cpu, h_C_shared, N);
        printf("Verification (shared): %s\n", correct ? "PASSED" : "FAILED");
        printf("\nSpeedup (naive):       %.1fx\n", timeCPU / timeGPU);
        printf("Speedup (shared):      %.1fx\n", timeCPU / timeShared);
    } else {
        bool correct = verifyResult(h_C_gpu, h_C_shared, N);
        printf("\nGPU naive vs shared:   %s\n", correct ? "MATCH" : "MISMATCH");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_C_shared);

    return 0;
}

