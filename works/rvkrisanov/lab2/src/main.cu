#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <functional>
#include <random>
#include <vector>
#include "../include/matrix_multiply.h"
#include "../include/error_check.cuh"

void initialize_matrix(float* matrix, const int rows, const int columns, std::mt19937& generator)
{
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            matrix[row * columns + column] = distribution(generator);
        }
    }
}

bool compare_matrices(const float* matrix_a, const float* matrix_b, const int rows, const int columns, const float epsilon = 1e-3f)
{
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            const int index = row * columns + column;
            if (fabsf(matrix_a[index] - matrix_b[index]) > epsilon)
            {
                return false;
            }
        }
    }
    return true;
}

double measure_time_microseconds(std::function<void()> function)
{
    auto start = std::chrono::high_resolution_clock::now();
    function();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

void make_benchmark(int matrix_size)
{
    const int rows_a = matrix_size;
    const int cols_a = matrix_size;
    const int cols_b = matrix_size;

    const size_t matrix_a_byte_size = rows_a * cols_a * sizeof(float);
    const size_t matrix_b_byte_size = cols_a * cols_b * sizeof(float);
    const size_t matrix_result_byte_size = rows_a * cols_b * sizeof(float);

    float *host_matrix_a, *host_matrix_b;
    float *host_result_cpu, *host_result_basic, *host_result_tiled;
    
    CUDA_CHECK(cudaMallocHost(&host_matrix_a, matrix_a_byte_size));
    CUDA_CHECK(cudaMallocHost(&host_matrix_b, matrix_b_byte_size));
    CUDA_CHECK(cudaMallocHost(&host_result_cpu, matrix_result_byte_size));
    CUDA_CHECK(cudaMallocHost(&host_result_basic, matrix_result_byte_size));
    CUDA_CHECK(cudaMallocHost(&host_result_tiled, matrix_result_byte_size));

    std::mt19937 generator(0);
    initialize_matrix(host_matrix_a, rows_a, cols_a, generator);
    initialize_matrix(host_matrix_b, cols_a, cols_b, generator);

    float *device_matrix_a, *device_matrix_b;
    float *device_result_basic, *device_result_tiled;
    
    CUDA_CHECK(cudaMalloc(&device_matrix_a, matrix_a_byte_size));
    CUDA_CHECK(cudaMalloc(&device_matrix_b, matrix_b_byte_size));
    CUDA_CHECK(cudaMalloc(&device_result_basic, matrix_result_byte_size));
    CUDA_CHECK(cudaMalloc(&device_result_tiled, matrix_result_byte_size));

    cudaStream_t stream_basic, stream_tiled;
    CUDA_CHECK(cudaStreamCreate(&stream_basic));
    CUDA_CHECK(cudaStreamCreate(&stream_tiled));

    cudaEvent_t start_basic, stop_basic, start_tiled, stop_tiled;
    CUDA_CHECK(cudaEventCreate(&start_basic));
    CUDA_CHECK(cudaEventCreate(&stop_basic));
    CUDA_CHECK(cudaEventCreate(&start_tiled));
    CUDA_CHECK(cudaEventCreate(&stop_tiled));

    printf("Benchmarking Matrix %dx%d...\n", matrix_size, matrix_size);

    CUDA_CHECK(cudaMemcpy(device_matrix_a, host_matrix_a, matrix_a_byte_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_matrix_b, host_matrix_b, matrix_b_byte_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_basic, stream_basic));
    launch_matrix_multiply_basic(device_matrix_a, device_matrix_b, device_result_basic, rows_a, cols_a, cols_b, stream_basic);
    CUDA_CHECK(cudaMemcpyAsync(host_result_basic, device_result_basic, matrix_result_byte_size, cudaMemcpyDeviceToHost, stream_basic));
    CUDA_CHECK(cudaEventRecord(stop_basic, stream_basic));

    CUDA_CHECK(cudaEventRecord(start_tiled, stream_tiled));
    launch_matrix_multiply_tiled(device_matrix_a, device_matrix_b, device_result_tiled, rows_a, cols_a, cols_b, stream_tiled);
    CUDA_CHECK(cudaMemcpyAsync(host_result_tiled, device_result_tiled, matrix_result_byte_size, cudaMemcpyDeviceToHost, stream_tiled));
    CUDA_CHECK(cudaEventRecord(stop_tiled, stream_tiled));

    double cpu_time_us = measure_time_microseconds([&]() {
        matrix_multiply_cpu(host_matrix_a, host_matrix_b, host_result_cpu, rows_a, cols_a, cols_b);
    });

    CUDA_CHECK(cudaStreamSynchronize(stream_basic));
    CUDA_CHECK(cudaStreamSynchronize(stream_tiled));

    float ms_basic = 0, ms_tiled = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_basic, start_basic, stop_basic));
    CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start_tiled, stop_tiled));

    bool basic_correct = compare_matrices(host_result_cpu, host_result_basic, rows_a, cols_b);
    bool tiled_correct = compare_matrices(host_result_cpu, host_result_tiled, rows_a, cols_b);

    double ms_basic_us = ms_basic * 1000.0;
    double ms_tiled_us = ms_tiled * 1000.0;

    printf("  CPU:        %8.0f us\n", cpu_time_us);
    printf("  CUDA Basic: %8.0f us (%.2fx speedup) [%s]\n", ms_basic_us, cpu_time_us / ms_basic_us, basic_correct ? "OK" : "FAIL");
    printf("  CUDA Tiled: %8.0f us (%.2fx speedup) [%s]\n", ms_tiled_us, cpu_time_us / ms_tiled_us, tiled_correct ? "OK" : "FAIL");

    CUDA_CHECK(cudaFreeHost(host_matrix_a));
    CUDA_CHECK(cudaFreeHost(host_matrix_b));
    CUDA_CHECK(cudaFreeHost(host_result_cpu));
    CUDA_CHECK(cudaFreeHost(host_result_basic));
    CUDA_CHECK(cudaFreeHost(host_result_tiled));

    CUDA_CHECK(cudaFree(device_matrix_a));
    CUDA_CHECK(cudaFree(device_matrix_b));
    CUDA_CHECK(cudaFree(device_result_basic));
    CUDA_CHECK(cudaFree(device_result_tiled));

    CUDA_CHECK(cudaEventDestroy(start_basic));
    CUDA_CHECK(cudaEventDestroy(stop_basic));
    CUDA_CHECK(cudaEventDestroy(start_tiled));
    CUDA_CHECK(cudaEventDestroy(stop_tiled));

    CUDA_CHECK(cudaStreamDestroy(stream_basic));
    CUDA_CHECK(cudaStreamDestroy(stream_tiled));
}

int main(int argc, char* argv[])
{
    std::vector<int> benchmark_sizes = {64, 128, 256, 512, 1024};

    if (argc > 1)
    {
        int size = atoi(argv[1]);
        make_benchmark(size);
    }
    else
    {
        for (int size : benchmark_sizes)
        {
            make_benchmark(size);
            printf("\n");
        }
    }

    return 0;
}
