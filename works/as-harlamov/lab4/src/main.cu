#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "sort.h"
#include "utils.h"

template<typename T>
void host_sequential_sort(T* output_buffer, T* input_buffer, size_t element_count) {
    for (size_t idx = 0; idx < element_count; idx++) {
        output_buffer[idx] = input_buffer[idx];
    }

    std::sort(output_buffer, output_buffer + element_count);
}

template<typename T>
void benchmark_comparison(T* host_input, unsigned int element_count) {
    std::clock_t timer_start;

    T* host_output_cpu = new T[element_count];

    timer_start = std::clock();
    host_sequential_sort(host_output_cpu, host_input, element_count);
    double cpu_time = (std::clock() - timer_start) / (double)CLOCKS_PER_SEC;
    std::cout << "CPU time: " << cpu_time << "s" << std::endl;

    T* device_input;
    T* device_output;
    CUDA_CHECK(cudaMalloc(&device_input, sizeof(T) * element_count));
    CUDA_CHECK(cudaMalloc(&device_output, sizeof(T) * element_count));
    CUDA_CHECK(cudaMemcpy(device_input, host_input, sizeof(T) * element_count, cudaMemcpyHostToDevice));
    timer_start = std::clock();
    radix_sort(device_output, device_input, element_count);
    double gpu_time = (std::clock() - timer_start) / (double)CLOCKS_PER_SEC;
    std::cout << "GPU time: " << gpu_time << "s" << std::endl;

    T* host_output_gpu = new T[element_count];
    CUDA_CHECK(cudaMemcpy(host_output_gpu, device_output, sizeof(T) * element_count, cudaMemcpyDeviceToHost));

    T* device_input_thrust;
    CUDA_CHECK(cudaMalloc(&device_input_thrust, sizeof(T) * element_count));
    CUDA_CHECK(cudaMemcpy(device_input_thrust, host_input, sizeof(T) * element_count, cudaMemcpyHostToDevice));
    
    timer_start = std::clock();
    thrust::sort(thrust::device, device_input_thrust, device_input_thrust + element_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    double thrust_time = (std::clock() - timer_start) / (double)CLOCKS_PER_SEC;
    std::cout << "Thrust time: " << thrust_time << "s" << std::endl;
    std::cout << "GPU / CPU: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Thrust / GPU: " << gpu_time / thrust_time << "x" << std::endl;

    CUDA_CHECK(cudaFree(device_input_thrust));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));


    bool is_correct = true;
    for (unsigned int i = 0; i < element_count; i++) {
        if (host_output_cpu[i] != host_output_gpu[i]) {
            is_correct = false;
            break;
        }
    }

    if (is_correct) {
        std::cout << "Sorting correctness: PASSED" << std::endl;
    } else {
        std::cout << "Sorting correctness: FAILED" << std::endl;
    }

    delete[] host_output_gpu;
    delete[] host_output_cpu;
}

template<typename T>
void run_tests_for_type(const char* type_name) {
    std::cout << "=================================" << std::endl;
    std::cout << "Testing " << type_name << " (" << sizeof(T) * 8 << "-bit integers)" << std::endl;
    std::cout << "=================================" << std::endl;

    unsigned int test_sizes[] = {100000, 500000, 1000000};
    const int num_tests = 3;

    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        unsigned int element_count = test_sizes[test_idx];
        std::cout << "\nArray size: " << element_count << std::endl;

        T* host_random_input = new T[element_count];
        for (int elem_idx = 0; elem_idx < element_count; elem_idx++) {
            host_random_input[elem_idx] = (T)((rand() % element_count) - (element_count / 2));
        }

        benchmark_comparison(host_random_input, element_count);
        std::cout << std::endl;

        delete[] host_random_input;
    }
}

int main() {
    srand(1);
    run_tests_for_type<int>("int32_t");    
    std::cout << "\n\n";
    run_tests_for_type<int64_t>("int64_t");
}