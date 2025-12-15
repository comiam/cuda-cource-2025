#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define TILE_SIZE 32
#define BLOCK_SIZE 32
#define RANDOM_SEED 666

#define CUDA_CHECK(err)                                                                    \
    do {                                                                                   \
        cudaError_t _e = (err);                                                            \
        if (_e != cudaSuccess) {                                                           \
            std::printf("%s in %s at line %d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

extern "C" void mat_mul_cpu(float* A, float* B, float* C, int N1, int N2, int N3) {
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N3; j++) {
            float value = 0.0f;
            for (int k = 0; k < N2; k++) {
                value += A[i * N2 + k] * B[k * N3 + j];
            }
            C[i * N3 + j] = value;
        }
    }
}

__global__ void mat_mul_kernel(const float* A, const float* B, float* C, int N1, int N2, int N3) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N1 && j < N3) {
        float value = 0.0f;
        for (int k = 0; k < N2; k++) {
            value += A[i * N2 + k] * B[k * N3 + j];
        }
        C[i * N3 + j] = value;
    }
}

void mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3, float* elapsed_ms) {
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, static_cast<size_t>(N1) * N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, static_cast<size_t>(N2) * N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, static_cast<size_t>(N1) * N3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, static_cast<size_t>(N1) * N2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, static_cast<size_t>(N2) * N3 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid(static_cast<unsigned>(std::ceil(N3 / 32.0f)),
                  static_cast<unsigned>(std::ceil(N1 / 32.0f)),
                  1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C, d_C, static_cast<size_t>(N1) * N3 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (elapsed_ms) {
        *elapsed_ms = kernel_ms;
    }
}

__global__ void tiled_mat_mul_kernel(const float* A, const float* B, float* C, int N1, int N2, int N3) {
    assert(TILE_SIZE == blockDim.x);
    assert(TILE_SIZE == blockDim.y);

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = TILE_SIZE * by + ty;
    int j = TILE_SIZE * bx + tx;

    __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

    float value = 0.0f;
    int phases = static_cast<int>(std::ceil(static_cast<float>(N2) / TILE_SIZE));

    for (int phase = 0; phase < phases; phase++) {
        int aCol = phase * TILE_SIZE + tx;
        int bRow = phase * TILE_SIZE + ty;

        sh_A[ty][tx] = (i < N1 && aCol < N2) ? A[i * N2 + aCol] : 0.0f;
        sh_B[ty][tx] = (bRow < N2 && j < N3) ? B[bRow * N3 + j] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        __syncthreads();
    }

    if (i < N1 && j < N3) {
        C[i * N3 + j] = value;
    }
}

void tiled_mat_mul_gpu(float* A, float* B, float* C, int N1, int N2, int N3, float* elapsed_ms) {
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, static_cast<size_t>(N1) * N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, static_cast<size_t>(N2) * N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, static_cast<size_t>(N1) * N3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, static_cast<size_t>(N1) * N2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, static_cast<size_t>(N2) * N3 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dim_block(TILE_SIZE, TILE_SIZE, 1);
    dim3 dim_grid(static_cast<unsigned>(std::ceil(N3 / static_cast<float>(TILE_SIZE))),
                  static_cast<unsigned>(std::ceil(N1 / static_cast<float>(TILE_SIZE))),
                  1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C, d_C, static_cast<size_t>(N1) * N3 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (elapsed_ms) {
        *elapsed_ms = kernel_ms;
    }
}

int main(int argc, char** argv) {
    int N1 = 1024, N2 = 1024, N3 = 1024;
    if (argc >= 4) {
        N1 = std::atoi(argv[1]);
        N2 = std::atoi(argv[2]);
        N3 = std::atoi(argv[3]);
    }

    std::vector<float> A(static_cast<size_t>(N1) * N2);
    std::vector<float> B(static_cast<size_t>(N2) * N3);
    std::vector<float> C_cpu(static_cast<size_t>(N1) * N3);
    std::vector<float> C_gpu(static_cast<size_t>(N1) * N3);
    std::vector<float> C_tiled(static_cast<size_t>(N1) * N3);

    std::mt19937 rng(RANDOM_SEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    mat_mul_cpu(A.data(), B.data(), C_cpu.data(), N1, N2, N3);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    float gpu_basic_ms = 0.0f;
    float gpu_tiled_ms = 0.0f;
    mat_mul_gpu(A.data(), B.data(), C_gpu.data(), N1, N2, N3, &gpu_basic_ms);
    tiled_mat_mul_gpu(A.data(), B.data(), C_tiled.data(), N1, N2, N3, &gpu_tiled_ms);

    std::printf("Matrix size: %dx%d\n", N1, N3);
    std::printf("CPU time: %.3f s\n", cpu_ms / 1000.0);
    std::printf("GPU time (basic): %.3f s\n", gpu_basic_ms / 1000.0);
    std::printf("GPU time (tiled): %.3f s\n", gpu_tiled_ms / 1000.0);

    if (gpu_basic_ms > 0.0f) {
        std::printf("Speed UP: ~%.1fx\n", (cpu_ms / 1000.0) / (gpu_basic_ms / 1000.0));
    }
    if (gpu_tiled_ms > 0.0f) {
        std::printf("Speed UP (tiled): ~%.1fx\n", (cpu_ms / 1000.0) / (gpu_tiled_ms / 1000.0));
    }

    return 0;
}

