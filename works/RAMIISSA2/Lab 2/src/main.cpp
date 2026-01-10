#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#include <cuda_runtime.h>

#include "matmul_cpu.hpp"
extern "C" void matmul_naive(const float*, const float*, float*, int, int, int);
extern "C" void matmul_tiled(const float*, const float*, float*, int, int, int, int);

bool compare(const float* C1, const float* C2, int M, int K, float eps = 1e-3f)
{
    for (int i = 0; i < M * K; i++) {
        float a = C1[i], b = C2[i];
        if (std::fabs(a - b) > eps) {
            // print first mismatch for debugging
            std::cout << "Mismatch at idx " << i << ": CPU=" << a << ", GPU=" << b << "\n";
            return false;
        }
    }
    return true;
}

void fill_rand(std::vector<float>& v, unsigned seed = 0)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

void run_case(int M, int N, int K)
{
    std::cout << "Test M=" << M << " N=" << N << " K=" << K << "\n";

    std::vector<float> A(M * N), B(N * K);
    std::vector<float> C_cpu(M * K), C_naive(M * K), C_tiled(M * K);

    fill_rand(A, 123);
    fill_rand(B, 456);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ===== GPU warm-up (not timed) =====
    matmul_naive(A.data(), B.data(), C_naive.data(), M, N, K);
    matmul_tiled(A.data(), B.data(), C_tiled.data(), M, N, K, 0);
    cudaDeviceSynchronize();


    // Naive GPU
    t0 = std::chrono::high_resolution_clock::now();
    matmul_naive(A.data(), B.data(), C_naive.data(), M, N, K);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok_naive = compare(C_cpu.data(), C_naive.data(), M, K, 1e-3f);


    // Tiled GPU
    t0 = std::chrono::high_resolution_clock::now();
    matmul_tiled(A.data(), B.data(), C_tiled.data(), M, N, K, 0);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double tiled_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok_tiled = compare(C_cpu.data(), C_tiled.data(), M, K, 1e-2f);


    // Print results
    std::cout << "CPU:        " << cpu_ms << " ms\n";
    std::cout << "Naive GPU:  " << naive_ms << " ms   " << (ok_naive ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "Tiled GPU:  " << tiled_ms << " ms   " << (ok_tiled ? "[OK]" : "[FAIL]") << "\n\n";
}

int main()
{
    // small correctness tests
    run_case(4, 4, 4);
    run_case(32, 32, 32);

    // rectangular edge cases
    run_case(37, 21, 19);
    run_case(128, 64, 23);

    // larger sizes (if your GPU has enough memory)
    run_case(512, 512, 512);
    run_case(1024, 1024, 1024);

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}
