#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdint.h>
#include <climits>
#include <random>
#include <type_traits>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <set>
#include "radix_sort.cuh"
#include "error_check.cuh"

using namespace std;


template<typename T>
vector<T> generateRandomArray(size_t length) {
    vector<T> result;
    result.reserve(length);

    random_device rd;
    mt19937 gen(rd());

    T min_val, max_val;
    if constexpr(is_same_v<T, uint32_t>) {
        min_val = 0;
        max_val = UINT32_MAX;
    }
    else if constexpr(is_same_v<T, uint64_t>) {
        min_val = 0;
        max_val = UINT64_MAX;
    }
    else if constexpr(is_same_v<T, int32_t>) {
        min_val = INT32_MIN;
        max_val = INT32_MAX;
    }
    else if constexpr(is_same_v<T, int64_t>) {
        min_val = INT64_MIN;
        max_val = INT64_MAX;
    }

    uniform_int_distribution<T> dis(min_val, max_val);
    for (size_t i = 0; i < length; i++) {
        result.push_back(dis(gen));
    }

    return result;
}


template<typename T>
void testSort(const vector<T>& testData, const char* typeName) {
    int n = testData.size();

    printf("Benchmark: %s[%d] ", typeName, n);

    // CPU sort with std::sort
    vector<T> cpuData = testData;
    auto cpuStart = chrono::high_resolution_clock::now();
    sort(cpuData.begin(), cpuData.end());
    chrono::duration<double> cpuElapsed = chrono::high_resolution_clock::now() - cpuStart;
    double cpuTime = cpuElapsed.count();

    // Allocate GPU memory
    T* d_input;
    T* d_output;
    T* d_thrust_input;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));
    cudaMalloc(&d_thrust_input, n * sizeof(T));

    cudaMemcpy(d_input, testData.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_thrust_input, testData.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Custom GPU Radix Sort
    auto gpuStart = chrono::high_resolution_clock::now();
    if constexpr(is_same_v<T, uint32_t>) {
        radixSort_uint32(d_input, d_output, n);
    }
    else if constexpr(is_same_v<T, int32_t>) {
        radixSort_int32(d_input, d_output, n);
    }
    else if constexpr(is_same_v<T, uint64_t>) {
        radixSort_uint64(d_input, d_output, n);
    }
    else if constexpr(is_same_v<T, int64_t>) {
        radixSort_int64(d_input, d_output, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    chrono::duration<double> gpuElapsed = chrono::high_resolution_clock::now() - gpuStart;
    double gpuTime = gpuElapsed.count();

    // Thrust::sort
    thrust::device_ptr<T> d_thrust_ptr = thrust::device_pointer_cast(d_thrust_input);
    auto thrustStart = chrono::high_resolution_clock::now();
    thrust::sort(d_thrust_ptr, d_thrust_ptr + n);
    CUDA_CHECK(cudaDeviceSynchronize());
    chrono::duration<double> thrustElapsed = chrono::high_resolution_clock::now() - thrustStart;
    double thrustTime = thrustElapsed.count();

    // Copy results back
    vector<T> gpuResult(n);
    vector<T> thrustResult(n);
    cudaMemcpy(gpuResult.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(thrustResult.data(), d_thrust_input, n * sizeof(T), cudaMemcpyDeviceToHost);

    // Verify correctness
    bool radixCorrect = true;
    bool thrustCorrect = true;
    for (int i = 0; i < n; i++) {
        if (cpuData[i] != gpuResult[i]) {
            radixCorrect = false;
        }
        if (cpuData[i] != thrustResult[i]) {
            thrustCorrect = false;
        }
        if (!radixCorrect && !thrustCorrect) break;
    }
    
    printf("%s\n", (radixCorrect && thrustCorrect) ? "OK" : "FAILED");
    if (!radixCorrect) printf("  WARNING: Radix Sort result incorrect!\n");
    if (!thrustCorrect) printf("  WARNING: Thrust Sort result incorrect!\n");
    
    printf("Time: CPU=%.5fs, GPU Radix=%.5fs, GPU Thrust=%.5fs\n", 
           cpuTime, gpuTime, thrustTime);
    printf("Speedup: Radix vs CPU=%.2fx, Thrust vs CPU=%.2fx, Radix vs Thrust=%.2fx\n\n",
           cpuTime / gpuTime, cpuTime / thrustTime, thrustTime / gpuTime);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_thrust_input);
}


int main() {
    set<int> arraySizes = {1000, 100000, 5000000, 10000000};

    for (const auto& size : arraySizes){
        vector<int32_t> testData32 = generateRandomArray<int32_t>(size);
        testSort<int32_t>(testData32, "int32_t");

        vector<int64_t> testData64 = generateRandomArray<int64_t>(size);
        testSort<int64_t>(testData64, "int64_t");

    }

    int uint_size = 5000000;
    vector<uint32_t> testData32 = generateRandomArray<uint32_t>(uint_size);
    testSort<uint32_t>(testData32, "uint32_t");

    vector<uint64_t> testData64 = generateRandomArray<uint64_t>(uint_size);
    testSort<uint64_t>(testData64, "uint64_t");

    return 0;
}