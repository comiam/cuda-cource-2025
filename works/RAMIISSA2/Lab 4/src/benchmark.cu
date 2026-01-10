#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <bitset>
#include "utils.h"
#include "cpu_sort.h"


void thrust_sort_gpu(const std::vector<int32_t>& input, std::vector<int32_t>& output, float& gpu_time_ms);
void radix_sort_gpu(const std::vector<int32_t>& input, std::vector<int32_t>& output, float& gpu_time_ms);
void radix_sort_gpu_opt(const std::vector<int32_t>& input, std::vector<int32_t>& output, float& gpu_time_ms);

void run_benchmark(size_t n) {
    std::cout << "\nArray size: " << n << std::endl;

    auto data = generate_random_ints(n);

    // std::sort
    auto std_data = data;

    auto std_start = std::chrono::high_resolution_clock::now();
    cpu_sort_std(std_data);
    auto std_end = std::chrono::high_resolution_clock::now();

    double std_time_ms =
        std::chrono::duration<double, std::milli>(std_end - std_start).count();

    std::cout << "CPU std::sort time: " << std_time_ms << " ms" << std::endl;

    // qsort
    auto qsort_data = data;

    auto qsort_start = std::chrono::high_resolution_clock::now();
    cpu_sort_qsort(qsort_data);
    auto qsort_end = std::chrono::high_resolution_clock::now();

    double qsort_time_ms =
        std::chrono::duration<double, std::milli>(qsort_end - qsort_start).count();

    std::cout << "CPU qsort time: " << qsort_time_ms << " ms" << std::endl;

    // thrust::sort
    std::vector<int32_t> gpu_thrust_data;
    float gpu_thrust_time_ms = 0.0f;

    thrust_sort_gpu(data, gpu_thrust_data, gpu_thrust_time_ms);

    std::cout << "\nGPU thrust::sort time: " << gpu_thrust_time_ms << " ms" << std::endl;

    // thrust correctness check
    bool thrust_correct = std::equal(std_data.begin(),
        std_data.end(),
        gpu_thrust_data.begin());

    std::cout << "GPU thrust::sort Correctness check: " << (thrust_correct ? "PASSED" : "FAILED") << std::endl;

    if (!thrust_correct) {
        std::cerr << "ERROR: GPU thrust sort result does not match CPU result\n";
        std::exit(EXIT_FAILURE);
    }

    // radix sort
    std::vector<int32_t> gpu_radix_data;
    float gpu_radix_time_ms = 0.0f;

    radix_sort_gpu(data, gpu_radix_data, gpu_radix_time_ms);

    std::cout << "\nGPU Radix Sort time: " << gpu_radix_time_ms << " ms" << std::endl;

    // radix correctness check
    bool radix_correct = std::equal(std_data.begin(), std_data.end(), gpu_radix_data.begin());
    std::cout << "GPU Radix Sort Correctness check: " << (radix_correct ? "PASSED" : "FAILED") << std::endl;

    if (!radix_correct) {
        std::cerr << "ERROR: GPU radix sort result does not match CPU result\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "---------------------------------------------------------------------";
}

void run_all() {
    run_benchmark(100000);
    run_benchmark(1000000);
    run_benchmark(10000000);
    run_benchmark(100000000);
}
