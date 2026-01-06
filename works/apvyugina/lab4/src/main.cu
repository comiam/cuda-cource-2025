#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdint.h>
#include <cuda_runtime.h>
#include "radix_sort.h"

using namespace std;

template<typename T>
void testSort(const vector<T>& testData, const char* typeName) {
    int n = testData.size();
    
    printf("\n=== Testing %s ===\n", typeName);
    printf("Array size: %d\n", n);
    
    // CPU sort with sort
    vector<T> cpuData = testData;
    auto cpuStart = chrono::high_resolution_clock::now();
    sort(cpuData.begin(), cpuData.end());
    auto cpuEnd = chrono::high_resolution_clock::now();
    auto cpuTime = chrono::duration_cast<chrono::microseconds>(cpuEnd - cpuStart).count();
    
    // GPU sort
    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));
    
    // Copy input to device (not timed)
    cudaMemcpy(d_input, testData.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    
    // Synchronize before timing
    cudaDeviceSynchronize();
    
    // Time only the sorting (not data transfer)
    auto gpuStart = chrono::high_resolution_clock::now();
    radixSort(d_input, d_output, n);
    cudaDeviceSynchronize();
    auto gpuEnd = chrono::high_resolution_clock::now();
    auto gpuTime = chrono::duration_cast<chrono::microseconds>(gpuEnd - gpuStart).count();
    
    // Copy result back (not timed)
    vector<T> gpuResult(n);
    cudaMemcpy(gpuResult.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (cpuData[i] != gpuResult[i]) {
            correct = false;
            if constexpr (sizeof(T) == 4) {
                printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, 
                       (int32_t)cpuData[i], (int32_t)gpuResult[i]);
            } else {
                printf("Mismatch at index %d: CPU=%ld, GPU=%ld\n", i, 
                       (int64_t)cpuData[i], (int64_t)gpuResult[i]);
            }
            break;
        }
    }
    
    printf("CPU sort time: %ld microseconds\n", cpuTime);
    printf("GPU sort time: %ld microseconds (excluding data transfer)\n", gpuTime);
    printf("Speedup: %.2fx\n", (double)cpuTime / gpuTime);
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    if (n <= 20) {
        printf("Sorted array: ");
        for (int i = 0; i < n; i++) {
            if constexpr (sizeof(T) == 4) {
                printf("%d ", (int32_t)gpuResult[i]);
            } else {
                printf("%ld ", (int64_t)gpuResult[i]);
            }
        }
        printf("\n");
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Test with int32_t - small array
    vector<uint32_t> testData32 = {170, 45, 75, 90, 2, 802, 24, 66, 7, 100, 127, 25, 34, 92, 9, 67, 578};
    testSort<uint32_t>(testData32, "int32_t (small)");
    
    // Test with int32_t - larger array
    vector<uint32_t> testData32Large;
    for (int i = 1000000; i > 0; i--) {
        testData32Large.push_back(i * 7 + 13);
    }
    testSort<uint32_t>(testData32Large, "int32_t (1000 elements)");
    
    // Test with int64_t
    // vector<int64_t> testData64 = {170LL, 45LL, 75LL, 90LL, 2LL, 802LL, 24LL, 66LL, 7LL, 100LL};
    // testSort<int64_t>(testData64, "int64_t (small)");
    
    // // Test with int64_t - larger array
    // vector<int64_t> testData64Large;
    // for (int i = 1000; i > 0; i--) {
    //     testData64Large.push_back((int64_t)i * 7 + 13);
    // }
    // testSort<int64_t>(testData64Large, "int64_t (1000 elements)");
    
    return 0;
}

