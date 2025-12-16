#include <cuda_runtime.h>
#include <stdio.h>

// Parameterizable tile size: change to 16 or 32 to experiment.
// You can also pass TILE_DIM via template or a define.
#ifndef TILE_DIM
#define TILE_DIM 32
#endif

// Tiled/shared memory matrix multiplication kernel.
// A: M x N
// B: N x K
// C: M x K
// We use padding in shared memory to reduce bank conflicts: stride = TILE_DIM + 1
__global__
void matmul_tiled_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Row and column of C this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory tiles with +1 padding to reduce bank conflicts
    __shared__ float As[TILE_DIM][TILE_DIM + 1];
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];

    float acc = 0.0f;

    // Number of tiles along the K / N dimension
    int fullTiles = N / TILE_DIM;          // tiles with no bounds issues
    int remainder = N % TILE_DIM;          // edge tile exists if > 0

    /* ============================================================
       Path 1: FULL tiles — no bounds checks at all
       ============================================================ */
    for (int t = 0; t < fullTiles; ++t)
    {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        As[ty][tx] = A[row * N + t * TILE_DIM + tx];
        Bs[ty][tx] = B[(t * TILE_DIM + ty) * K + col];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k)
            acc += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    /* ============================================================
       Path 2: EDGE tile — bounds checks (executed once)
       ============================================================ */
    if (remainder > 0)
    {
        int t = fullTiles;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int aCol = t * TILE_DIM + tx;
        int bRow = t * TILE_DIM + ty;

        As[ty][tx] = (row < M && aCol < N)
            ? A[row * N + aCol]
            : 0.0f;

        Bs[ty][tx] = (bRow < N && col < K)
            ? B[bRow * K + col]
            : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k)
            acc += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = acc;
}

// C-callable wrapper (use extern "C" if linking from C++ main compiled as C++)
extern "C"
void matmul_tiled(const float* A, const float* B, float* C,
    int M, int N, int K,
    int tile_dim_override)
{
    // allow runtime override of tile dim by selecting block dimension accordingly
    // But compiled tile size (TILE_DIM) is used for shared memory layout.
    // tile_dim_override is informative — set block dims to match TILE_DIM normally.
    const int tile = TILE_DIM;

    float* dA = nullptr, * dB = nullptr, * dC = nullptr;
    size_t bytesA = size_t(M) * N * sizeof(float);
    size_t bytesB = size_t(N) * K * sizeof(float);
    size_t bytesC = size_t(M) * K * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&dA, bytesA); if (err != cudaSuccess) { printf("cudaMalloc dA failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&dB, bytesB); if (err != cudaSuccess) { printf("cudaMalloc dB failed: %s\n", cudaGetErrorString(err)); cudaFree(dA); return; }
    err = cudaMalloc(&dC, bytesC); if (err != cudaSuccess) { printf("cudaMalloc dC failed: %s\n", cudaGetErrorString(err)); cudaFree(dA); cudaFree(dB); return; }

    cudaMemcpy(dA, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(tile, tile);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Optionally adjust shared-memory size (not required here because it is statically allocated)
    matmul_tiled_kernel << <grid, block >> > (dA, dB, dC, M, N, K);

    cudaMemcpy(C, dC, bytesC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

extern "C"
void matmul_tiled_launch_device(const float* dA, const float* dB, float* dC,
    int M, int N, int K, dim3 grid, dim3 block)
{
    matmul_tiled_kernel << <grid, block >> > (dA, dB, dC, M, N, K);
}