#include <cstdlib>
#include <cmath>
#include <ctime>


void fillMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyResult(const float* C_cpu, const float* C_gpu, int M, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C_cpu[i] - C_gpu[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void matMulCPU(const float* A, const float* B, float* C, int M, int K, int N) {
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
