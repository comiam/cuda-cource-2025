#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <type_traits>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "../include/radix_sort.h"
#include "../include/error_check.cuh"

template <typename T>
std::vector<T> generate_data(int size) {
    std::vector<T> data(size);
    std::mt19937_64 gen(0); 
    
    std::uniform_int_distribution<T> dist;
    if constexpr (std::is_same_v<T, int>) {
        dist = std::uniform_int_distribution<T>(INT32_MIN, INT32_MAX);
    } else {
        dist = std::uniform_int_distribution<T>(INT64_MIN, INT64_MAX);
    }
    
    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

template <typename T>
void verify_results(const std::vector<T>& cpu_ref, const std::vector<T>& gpu_result) {
    if (cpu_ref.size() != gpu_result.size()) {
        std::cerr << "[FAIL] Sizes differ!" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < cpu_ref.size(); ++i) {
        if (cpu_ref[i] != gpu_result[i]) {
            std::cerr << "[FAIL] Mismatch at index " << i 
                      << ": CPU=" << cpu_ref[i] 
                      << ", GPU=" << gpu_result[i] << std::endl;
            return;
        }
    }
    std::cout << "[OK] Results match!" << std::endl;
}

void warmup() {
    std::cout << "Warming up CUDA and Thrust..." << std::endl;
    thrust::device_vector<int> device_array(100);
    thrust::sort(device_array.begin(), device_array.end());
    cudaDeviceSynchronize();
    std::cout << "Warmup done." << std::endl;
}

template <typename T>
void run_benchmark_for_type(const std::string& type_name, const std::vector<int>& sizes) {
    std::cout << "================================================================" << std::endl;
    std::cout << "Running Benchmark for Type: " << type_name << " (" << sizeof(T) * 8 << "-bit)" << std::endl;
    std::cout << "================================================================" << std::endl;

    for (int n : sizes) {
        std::cout << "Size N=" << n << std::endl;
        
        auto host_data = generate_data<T>(n);
        auto host_data_cpu = host_data;
        auto host_data_thrust = host_data;
        
        auto start_cpu = std::chrono::high_resolution_clock::now();
        radix_sort_cpu(host_data_cpu);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        double cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
        
        T *device_input, *device_temp;
        CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&device_temp, n * sizeof(T)));
        
        int num_iterations = 5;
        double total_custom_us = 0;
        double total_thrust_us = 0;

        for(int i=0; i<num_iterations; ++i) {
            CUDA_CHECK(cudaMemcpy(device_input, host_data.data(), n * sizeof(T), cudaMemcpyHostToDevice));
            
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            CUDA_CHECK(cudaEventRecord(start));
            radix_sort_gpu(device_input, device_temp, n);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_custom_us += ms * 1000.0;
            
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        std::vector<T> gpu_result_custom(n);
        CUDA_CHECK(cudaMemcpy(gpu_result_custom.data(), device_input, n * sizeof(T), cudaMemcpyDeviceToHost));
        std::cout << "Verifying Custom Implementation: ";
        verify_results(host_data_cpu, gpu_result_custom);

        thrust::device_vector<T> device_thrust_vec = host_data_thrust; 
        
        for(int i=0; i<num_iterations; ++i) {
            device_thrust_vec = host_data_thrust; 
            
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            
            CUDA_CHECK(cudaEventRecord(start));
            thrust::sort(device_thrust_vec.begin(), device_thrust_vec.end());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_thrust_us += ms * 1000.0;
            
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        
        double avg_custom_us = total_custom_us / num_iterations;
        double avg_thrust_us = total_thrust_us / num_iterations;
        
        std::cout << "Avg Time CPU:           " << (long long)cpu_us << " us" << std::endl;
        std::cout << "Avg Time Custom GPU:    " << (long long)avg_custom_us << " us" << std::endl;
        std::cout << "Avg Time Thrust GPU:    " << (long long)avg_thrust_us << " us" << std::endl;
        std::cout << "Speedup vs CPU:     " << cpu_us / avg_custom_us << "x" << std::endl;
        std::cout << "Speedup vs Thrust:  " << avg_thrust_us / avg_custom_us << "x" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        
        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_temp));
    }
}

int main() {
    warmup();
    
    std::vector<int> sizes = {100000, 500000, 1000000};
    
    run_benchmark_for_type<int>("int32", sizes);
    
    std::cout << "\n\n";
    
    run_benchmark_for_type<long long>("int64", sizes);
    
    return 0;
}
