#ifndef KERNEL_LAUNCHER
#define KERNEL_LAUNCHER

#define BLOCK_DIM 16
#include <cuda_runtime.h>

template<typename KernelT>
double multiplyMatrices(
    KernelT multiplicationKernel,
    vector<vector<float>> &A, vector<vector<float>> &B, vector<vector<float>> &C
){
    int M = A.size();
    int N = A[0].size();
    int P = B[0].size();

    vector<float> flat_A = flattenMatrix(A);
    vector<float> flat_B = flattenMatrix(B);


    float *A_device, *B_device, *res_device;
    cudaMalloc(&A_device, M * N * sizeof(float));
    cudaMalloc(&B_device, N * P * sizeof(float));
    cudaMalloc(&res_device, M * P * sizeof(float));

    cudaMemcpy(A_device, flat_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, flat_B.data(), N * P * sizeof(float), cudaMemcpyHostToDevice);

    // Определяем размер блока и сетки
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);    

    // Запускаем ядро
    auto start_time = chrono::high_resolution_clock::now();
    multiplicationKernel<<<gridSize, blockSize>>>(A_device, B_device, res_device, M, N, P);
    chrono::duration<float> elapsed = chrono::high_resolution_clock::now() - start_time;


    cudaDeviceSynchronize();
    vector<float> result(M * P);
    cudaMemcpy(result.data(), res_device, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < P; ++j) {
            C[i][j] = result[i * P + j];
        }
    }

    // Освобождаем память устройства
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(res_device);

    return elapsed.count();
}

#endif