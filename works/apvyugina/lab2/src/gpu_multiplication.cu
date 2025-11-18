#include <cuda_runtime.h>
#include <vector>

#include "utils.h"
#include "kernel_launcher.h"

using namespace std;


__global__ 
void simpleMatrixMultiplication(float* A, float* B, float* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < P) {
        float sum = 0.0f;
        for(int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


__global__ 
void sharedMatrixMultiplication(float* A, float* B, float* C, int M, int N, int P, int matrix_size) {
    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (matrix_size + BLOCK_DIM - 1)/BLOCK_DIM; ++t) {
        if (row < M && threadIdx.x + t * BLOCK_DIM < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + threadIdx.x + t * BLOCK_DIM];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < P && threadIdx.y + t * BLOCK_DIM < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(threadIdx.y + t * BLOCK_DIM) * P + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_DIM; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < P) {
        C[row * P + col] = acc;
    }
}



int main(int argc, char* argv[]) {

    const int MATRIX_SIZE = stoi(argv[1]);

    init_random_generator();

    auto A = generateMatrix<float>(MATRIX_SIZE);
    auto B = generateMatrix<float>(MATRIX_SIZE);

    cout << "GPU" << endl;
    vector<vector<float>> res(MATRIX_SIZE, vector<float>(MATRIX_SIZE));
    double elapsed = multiplyMatrices(simpleMatrixMultiplication, A, B, res);
    cout << "Простое умножение заняло: " << elapsed << " секунд.\n";

    vector<vector<float>> shared_res(MATRIX_SIZE, vector<float>(MATRIX_SIZE));
    elapsed = multiplyMatrices(sharedMatrixMultiplication, A, B, shared_res);
    cout << "Умножение через shared_memory заняло: " << elapsed << " секунд.\n";

    bool areEqual = compareMatrices(res, shared_res);
    cout << "Матрицы равны: " << areEqual << endl;
    return 0;
}