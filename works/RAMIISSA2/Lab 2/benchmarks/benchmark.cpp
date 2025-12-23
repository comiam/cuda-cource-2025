// benchmarks/benchmark.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>
#include <iomanip>

#include <cuda_runtime.h>

#include "../src/matmul_cpu.hpp"

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

// wrappers in your CUDA files
extern "C" void matmul_naive(const float*, const float*, float*, int, int, int);
extern "C" void matmul_tiled(const float*, const float*, float*, int, int, int, int);

// device-only launch wrappers (added to the .cu files)
extern "C" void matmul_naive_launch_device(const float* dA, const float* dB, float* dC,
    int M, int N, int K, dim3 grid, dim3 block);
extern "C" void matmul_tiled_launch_device(const float* dA, const float* dB, float* dC,
    int M, int N, int K, dim3 grid, dim3 block);

static void fill_rand(std::vector<float>& v, unsigned seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

static void cuda_check(cudaError_t e, const char* msg = nullptr) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e);
        if (msg) std::cerr << " (" << msg << ")";
        std::cerr << std::endl;
        std::exit(1);
    }
}

struct Stats {
    double mean = 0, median = 0, minimum = 0;
};

static Stats summarize(std::vector<double>& v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (auto x : v) sum += x;
    s.mean = sum / v.size();
    s.minimum = v.front();
    if (v.size() % 2 == 1) s.median = v[v.size() / 2];
    else s.median = 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    return s;
}

int main(int argc, char** argv) {
    // Configuration
    const int repeats = 100;           // how many repeated measurements to run
    const bool measure_kernel_only = true;
    const std::string out_csv = "benchmarks/results_" + std::to_string(TILE_DIM) + ".csv";

    // List of test sizes (M,N,K). Edit or extend as needed.
    std::vector<std::tuple<int, int, int>> tests = {
        std::make_tuple(32, 32, 32),
        std::make_tuple(128, 128, 128),
        std::make_tuple(256, 256, 256),
        std::make_tuple(512, 512, 512),
        std::make_tuple(1024, 1024, 1024),
        std::make_tuple(512, 1024, 512),
        std::make_tuple(1024, 512, 1024),
        std::make_tuple(512, 1024, 256)
    };

    // Prepare CSV
    std::ofstream csv(out_csv);
    if (!csv.is_open()) {
        std::cerr << "Error: cannot open CSV file for writing!\n";
        return 1;
    }

    csv << "M,N,K,CPU_ms,Naive_full_ms,Naive_kernel_ms, Tiled_full_ms, Tiled_kernel_ms\n";

    csv << std::fixed << std::setprecision(3);
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Starting benchmarks (repeats=" << repeats << ") with Tile Dimention = " << TILE_DIM << ", CSV -> " << out_csv << "\n";

    for (auto [M, N, K] : tests) {
        std::cout << "=== Test " << M << " x " << N << " x " << K << " ===\n";

        size_t bytesA = size_t(M) * N * sizeof(float);
        size_t bytesB = size_t(N) * K * sizeof(float);
        size_t bytesC = size_t(M) * K * sizeof(float);

        std::vector<float> A(M * N), B(N * K), C_cpu(M * K), C_tmp(M * K);
        fill_rand(A, 1234);
        fill_rand(B, 5678);

        // CPU timing (single-thread)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "CPU: " << cpu_ms << " ms\n";
            // store placeholder; real csv row will be written after GPU measurements
        }

        // Warm-up full-call (alloc+copy+kernel+copyback) to initialize CUDA context
        matmul_naive(A.data(), B.data(), C_tmp.data(), M, N, K);
        matmul_tiled(A.data(), B.data(), C_tmp.data(), M, N, K, 0);
        cudaDeviceSynchronize();

        // Measure full-call times (wrapper functions that include malloc/copy)
        std::vector<double> naive_full_times, tiled_full_times;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_naive(A.data(), B.data(), C_tmp.data(), M, N, K);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            naive_full_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

            t0 = std::chrono::high_resolution_clock::now();
            matmul_tiled(A.data(), B.data(), C_tmp.data(), M, N, K, 0);
            cudaDeviceSynchronize();
            t1 = std::chrono::high_resolution_clock::now();
            tiled_full_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        Stats naive_full_stats = summarize(naive_full_times);
        Stats tiled_full_stats = summarize(tiled_full_times);

        std::cout << "Naive full (min/med/mean): " << naive_full_stats.minimum << " / " << naive_full_stats.median << " / " << naive_full_stats.mean << " ms\n";
        std::cout << "Tiled full (min/med/mean): " << tiled_full_stats.minimum << " / " << tiled_full_stats.median << " / " << tiled_full_stats.mean << " ms\n";

        // Measure kernel-only times
        double naive_kernel_min = 0, naive_kernel_median = 0, naive_kernel_mean = 0;
        double tiled_kernel_min = 0, tiled_kernel_median = 0, tiled_kernel_mean = 0;

        if (measure_kernel_only) {
            // Allocate device arrays once
            float* dA = nullptr, * dB = nullptr, * dC = nullptr;
            cuda_check(cudaMalloc(&dA, bytesA), "alloc dA");
            cuda_check(cudaMalloc(&dB, bytesB), "alloc dB");
            cuda_check(cudaMalloc(&dC, bytesC), "alloc dC");

            // Copy input once
            cuda_check(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice), "copy A");
            cuda_check(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice), "copy B");

            // choose block & grid like in your wrappers (block 16x16)
            dim3 block(16, 16);
            dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

            // Warmup device-only launches
            matmul_naive_launch_device(dA, dB, dC, M, N, K, grid, block);
            matmul_tiled_launch_device(dA, dB, dC, M, N, K, grid, block);
            cudaDeviceSynchronize();

            // Prepare vectors for raw times
            std::vector<double> naive_kernel_times, tiled_kernel_times;
            naive_kernel_times.reserve(repeats);
            tiled_kernel_times.reserve(repeats);

            // Use CUDA events for accurate kernel timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            for (int r = 0; r < repeats; ++r) {
                // Naive kernel-only
                cudaEventRecord(start);
                matmul_naive_launch_device(dA, dB, dC, M, N, K, grid, block);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float msNaive = 0.0f;
                cudaEventElapsedTime(&msNaive, start, stop);
                naive_kernel_times.push_back(msNaive);

                // Tiled kernel-only
                cudaEventRecord(start);
                matmul_tiled_launch_device(dA, dB, dC, M, N, K, grid, block);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float msTiled = 0.0f;
                cudaEventElapsedTime(&msTiled, start, stop);
                tiled_kernel_times.push_back(msTiled);
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // free device mem
            cudaFree(dA); cudaFree(dB); cudaFree(dC);

            Stats naive_kernel_stats = summarize(naive_kernel_times);
            Stats tiled_kernel_stats = summarize(tiled_kernel_times);

            naive_kernel_min = naive_kernel_stats.minimum;
            naive_kernel_median = naive_kernel_stats.median;
            naive_kernel_mean = naive_kernel_stats.mean;

            tiled_kernel_min = tiled_kernel_stats.minimum;
            tiled_kernel_median = tiled_kernel_stats.median;
            tiled_kernel_mean = tiled_kernel_stats.mean;

            std::cout << "Naive kernel (min/med/mean): " << naive_kernel_min << " / " << naive_kernel_median << " / " << naive_kernel_mean << " ms\n";
            std::cout << "Tiled kernel (min/med/mean): " << tiled_kernel_min << " / " << tiled_kernel_median << " / " << tiled_kernel_mean << " ms\n";
        }

        // Write CSV row: choose min for full-call and kernel times
        csv << M << "," << N << "," << K << ",";

        // For CPU we measured earlier only once; re-run quick CPU to get mean
        {
            std::vector<double> cpu_times;
            for (int r = 0; r < 3; ++r) {
                auto t0 = std::chrono::high_resolution_clock::now();
                matmul_cpu(A.data(), B.data(), C_tmp.data(), M, N, K);
                auto t1 = std::chrono::high_resolution_clock::now();
                cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            Stats s = summarize(cpu_times);
            csv << s.mean << ",";
        }

        csv << naive_full_stats.mean << "," << naive_kernel_mean << ",";
        csv << tiled_full_stats.mean << "," << tiled_kernel_mean << "\n";
        csv.flush();
    }

    csv.close();
    std::cout << "\nDone.";
    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}
