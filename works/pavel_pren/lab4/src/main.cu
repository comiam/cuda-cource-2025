#include "radix_sort.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>

using namespace std::chrono;

// класс для замера времени на CPU
class Timer {
private:
    high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

int compareInt(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

// сортировка на CPU
void cpuSort(int* data, int size) {
    qsort(data, size, sizeof(int), compareInt);
}

void generateRandomData(int* data, int size, int minVal, int maxVal) {
    for (int i = 0; i < size; i++) {
        data[i] = minVal + rand() % (maxVal - minVal + 1);
    }
}

void generateTestData(int* data, int size) {
    for (int i = 0; i < size; i++) {
        if (rand() % 2 == 0) {
            data[i] = rand() % 1000000;
        } else {
            data[i] = -(rand() % 1000000);
        }
    }
}

bool verifyResults(const int* data1, const int* data2, int size) {
    for (int i = 0; i < size; i++) {
        if (data1[i] != data2[i]) {
            printf("Mismatch at index %d: %d != %d\n", i, data1[i], data2[i]);
            return false;
        }
    }
    return true;
}

void runBenchmark(int size) {
    printf("\nRadix Sort Benchmark\n");
    printf("Array size: %d\n", size);
    printf("\n");
    
    // выделяем память для разных версий массива
    int* h_data = new int[size];
    int* h_cpu = new int[size];
    int* h_gpu = new int[size];
    int* h_thrust = new int[size];
    
    printf("init array with random data\n");
    srand(time(NULL));
    generateTestData(h_data, size);
    
    memcpy(h_cpu, h_data, size * sizeof(int));
    memcpy(h_gpu, h_data, size * sizeof(int));
    memcpy(h_thrust, h_data, size * sizeof(int));
    
    Timer timer;
    
    // тест CPU
    printf("Running CPU qsort\n");
    timer.start();
    cpuSort(h_cpu, size);
    double cpuTime = timer.stop();
    printf("CPU time: %.2f ms\n", cpuTime);
    
    // тест GPU
    printf("\nRunning GPU Radix Sort\n");
    
    int* d_gpu;
    CUDA_CHECK(cudaMalloc(&d_gpu, size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_gpu, h_gpu, size * sizeof(int), cudaMemcpyHostToDevice));
    
    radixSortInt32(d_gpu, size); // разогрев GPU
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(d_gpu, h_gpu, size * sizeof(int), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    radixSortInt32(d_gpu, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpuTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
    
    printf("GPU time: %.2f ms\n", gpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    
    CUDA_CHECK(cudaMemcpy(h_gpu, d_gpu, size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // тест Thrust
    printf("\nRunning Thrust sort\n");
    
    thrust::device_vector<int> d_thrust(h_thrust, h_thrust + size);
    
    CUDA_CHECK(cudaEventRecord(start));
    thrust::sort(d_thrust.begin(), d_thrust.end());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float thrustTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&thrustTime, start, stop));
    
    printf("Thrust sort time: %.2f ms\n", thrustTime);
    
    thrust::copy(d_thrust.begin(), d_thrust.end(), h_thrust);
    
    printf("\nVerifying\n");
    
    if (verifySorted(h_gpu, size)) {
        printf("Radix Sort: OK\n");
    } else {
        printf("Radix Sort: FAILED\n");
    }
    
    if (verifyResults(h_gpu, h_thrust, size)) {
        printf("Match with Thrust: OK\n");
    } else {
        printf("Match with Thrust: FAILED\n");
    }
    
    if (size <= 100) {
        printf("\nSorted array:\n");
        printArray(h_gpu, size);
    }
    
    delete[] h_data;
    delete[] h_cpu;
    delete[] h_gpu;
    delete[] h_thrust;
    CUDA_CHECK(cudaFree(d_gpu));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void runTests() {
    printf("\nRunning Tests\n\n");
    
    // тест 1: маленький массив
    {
        printf("Test 1: Small array (100 elements)\n");
        int size = 100;
        int* h_data = new int[size];
        generateTestData(h_data, size);
        
        int* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));
        
        radixSortInt32(d_data, size);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
        
        if (verifySorted(h_data, size)) {
            printf("Test 1: PASSED\n\n");
        } else {
            printf("Test 1: FAILED\n\n");
        }
        
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_data));
    }
    
    // тест 2: только отрицательные числа
    {
        printf("Test 2: All negative numbers (1000 elements)\n");
        int size = 1000;
        int* h_data = new int[size];
        for (int i = 0; i < size; i++) {
            h_data[i] = -(rand() % 1000000);
        }
        
        int* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));
        
        radixSortInt32(d_data, size);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
        
        if (verifySorted(h_data, size)) {
            printf("Test 2: PASSED\n\n");
        } else {
            printf("Test 2: FAILED\n\n");
        }
        
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_data));
    }
    
    // тест 3: уже отсортированный массив
    {
        printf("Test 3: Already sorted array (1000 elements)\n");
        int size = 1000;
        int* h_data = new int[size];
        for (int i = 0; i < size; i++) {
            h_data[i] = i - 500;
        }
        
        int* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));
        
        radixSortInt32(d_data, size);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
        
        if (verifySorted(h_data, size)) {
            printf("Test 3: PASSED\n\n");
        } else {
            printf("Test 3: FAILED\n\n");
        }
        
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_data));
    }
    
    // тест 4: обратно отсортированный
    {
        printf("Test 4: Reverse sorted array (1000 elements)\n");
        int size = 1000;
        int* h_data = new int[size];
        for (int i = 0; i < size; i++) {
            h_data[i] = 500 - i;
        }
        
        int* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));
        
        radixSortInt32(d_data, size);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
        
        if (verifySorted(h_data, size)) {
            printf("Test 4: PASSED\n\n");
        } else {
            printf("Test 4: FAILED\n\n");
        }
        
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_data));
    }
    
    printf("Tests complete\n\n");
}

int main(int argc, char** argv) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    int size = 1000000;
    
    if (argc > 1) {
        size = atoi(argv[1]);
        if (size <= 0) {
            fprintf(stderr, "Invalid array size: %s\n", argv[1]);
            return 1;
        }
    }
    
    if (size <= 10000) {
        runTests();
    }
    
    runBenchmark(size); // запускаем бенчмарк
    
    return 0;
}
