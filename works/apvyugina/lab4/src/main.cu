#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdint.h>
#include <random>
#include <type_traits>
#include <cuda_runtime.h>
#include <set>
#include "radix_sort.h"

using namespace std;

template<typename T>
vector<T> generateRandomArray(size_t length) {
    vector<T> result;
    result.reserve(length);
    
    random_device rd;
    mt19937 gen(rd());
    
    if constexpr(is_same_v<T, uint32_t>) {
        uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
        for (size_t i = 0; i < length; i++) {
            result.push_back(dis(gen));
        }
    }
    else if constexpr(is_same_v<T, uint64_t>) {
        uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
        for (size_t i = 0; i < length; i++) {
            result.push_back(dis(gen));
        }
    }
    
    return result;
}

template<typename T>
void testSort(const vector<T>& testData, const char* typeName) {
    int n = testData.size();
    
    printf("\n=== Benchmark: %s [%d] ===\n", typeName, n);

    // CPU sort with sort
    vector<T> cpuData = testData;
    auto cpuStart = chrono::high_resolution_clock::now();
    sort(cpuData.begin(), cpuData.end());
    chrono::duration<double> cpuElapsed = chrono::high_resolution_clock::now() - cpuStart;
    double cpuTime = cpuElapsed.count();

    // GPU sort
    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));
    
    cudaMemcpy(d_input, testData.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    auto gpuStart = chrono::high_resolution_clock::now();
    if constexpr(is_same_v<T, uint32_t>) {
        radixSort_int32(d_input, d_output, n);
    }
    else if constexpr(is_same_v<T, uint64_t>) {
        radixSort_int64(d_input, d_output, n);
    }

    cudaDeviceSynchronize();
    chrono::duration<double> gpuElapsed = chrono::high_resolution_clock::now() - gpuStart;
    double gpuTime = gpuElapsed.count();
    
    vector<T> gpuResult(n);
    cudaMemcpy(gpuResult.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (cpuData[i] != gpuResult[i]) {
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("Time: CPU=%.3fs, GPU=%.3fs\n", cpuTime, gpuTime);
    printf("Speedup: %.2fx\n", (double)cpuTime / gpuTime);
    
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    set<int> arraySizes = {100, 100000, 5000000, 10000000};
    
    for (const auto& size : arraySizes){
        vector<uint32_t> testData32 = generateRandomArray<uint32_t>(size);
        testSort<uint32_t>(testData32, "uint32_t");

        vector<uint64_t> testData64 = generateRandomArray<uint64_t>(size);
        testSort<uint64_t>(testData64, "uint64_t");

    }
    
    return 0;
}

