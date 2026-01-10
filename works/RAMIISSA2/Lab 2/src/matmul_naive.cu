#include <cuda_runtime.h>
#include <stdio.h>

__global__
void matmul_naive_kernel(const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = sum;
    }
}

extern "C"
void matmul_naive(const float* A, const float* B, float* C,
    int M, int N, int K)
{
    float* dA, * dB, * dC;

    size_t bytesA = M * N * sizeof(float);
    size_t bytesB = N * K * sizeof(float);
    size_t bytesC = M * K * sizeof(float);

    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    matmul_naive_kernel << <grid, block >> > (dA, dB, dC, M, N, K);

    cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C"
void matmul_naive_launch_device(const float* dA, const float* dB, float* dC,
    int M, int N, int K, dim3 grid, dim3 block)
{
    // Launch the kernel on already-allocated device pointers
    matmul_naive_kernel << <grid, block >> > (dA, dB, dC, M, N, K);
}