#include "matmul_cpu.hpp"

void matmul_cpu(const float* A, const float* B, float* C,
    int M, int N, int K)
{
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
