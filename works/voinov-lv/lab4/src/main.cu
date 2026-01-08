#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <utility>

#define CUDA_CHECK(call)                                                         \
do {                                                                             \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
        fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__,                \
                cudaGetErrorString(err__));                                      \
        std::abort();                                                            \
    }                                                                            \
} while(0)

template<typename T>
__global__ void preds_kernel(const T* __restrict__ input, 
                                 int* __restrict__ preds, int n, int bit_shift) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        int bit = (val >> bit_shift) & 1;
        preds[idx] = 1 - bit;
    }
}

template<typename T>
__global__ void scatter_kernel(const T* __restrict__ input, T* __restrict__ output, 
                               const int* __restrict__ scanned_preds, 
                               int n, int bit_shift, int total_zeros) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        T val = input[idx];
        int bit = (val >> bit_shift) & 1;
        int rank_if_0 = scanned_preds[idx];
        int rank_if_1 = (idx - rank_if_0) + total_zeros;
        int dest = (bit == 0) ? rank_if_0 : rank_if_1;

        output[dest] = val;
    }
}

__global__ void prescan_kernel(int *g_out, const int *g_in, int *g_block_sums, int n) {
    extern __shared__ int temp[]; 

    int thid = threadIdx.x;
    int offset = 1;
    int block_offset = blockIdx.x * (blockDim.x * 2);
    int ai = block_offset + (2 * thid);
    int bi = block_offset + (2 * thid) + 1;

    temp[2 * thid] = (ai < n) ? g_in[ai] : 0;
    temp[2 * thid + 1] = (bi < n) ? g_in[bi] : 0;

    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai_s = offset * (2 * thid + 1) - 1;
            int bi_s = offset * (2 * thid + 2) - 1;
            temp[bi_s] += temp[ai_s];
        }

        offset *= 2;
    }

    if (thid == 0) {
        if (g_block_sums != nullptr) {
            g_block_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
        }
        temp[2 * blockDim.x - 1] = 0;
    }

    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai_s = offset * (2 * thid + 1) - 1;
            int bi_s = offset * (2 * thid + 2) - 1;
            int t = temp[ai_s];
            temp[ai_s] = temp[bi_s];
            temp[bi_s] += t;
        }
    }
    
    __syncthreads();

    if (ai < n) g_out[ai] = temp[2 * thid];
    if (bi < n) g_out[bi] = temp[2 * thid + 1];
}

__global__ void add_block_sums_kernel(int *g_out, const int *g_block_sums, int n) {
    int block_offset = blockIdx.x * (blockDim.x * 2);
    int ai = block_offset + (2 * threadIdx.x);
    int bi = block_offset + (2 * threadIdx.x) + 1;

    int block_sum = g_block_sums[blockIdx.x];

    if (ai < n) g_out[ai] += block_sum;
    if (bi < n) g_out[bi] += block_sum;
}

#define SECTION_SIZE 1024

void exclusive_scan(int* d_out, int* d_in, int n) {
    int threadsPerBlock = SECTION_SIZE / 2;
    int numBlocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;

    if (numBlocks <= 1) {
        prescan_kernel<<<1, threadsPerBlock, SECTION_SIZE * sizeof(int)>>>(d_out, d_in, nullptr, n);
        return;
    }

    int* d_block_sums;
    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, numBlocks * sizeof(int)));

    int* d_scanned_block_sums;
    CUDA_CHECK(cudaMalloc((void**)&d_scanned_block_sums, numBlocks * sizeof(int)));

    prescan_kernel<<<numBlocks, threadsPerBlock, SECTION_SIZE * sizeof(int)>>>(d_out, d_in, d_block_sums, n);

    exclusive_scan(d_scanned_block_sums, d_block_sums, numBlocks);

    add_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_scanned_block_sums, n);

    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_scanned_block_sums));
}

template<typename T>
void radix_sort(T* h_data, int n) {
    const int BITS = sizeof(T) * 8;
    
    T *d_in, *d_out;
    int *d_preds, *d_scanned_preds;
    
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_preds, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scanned_preds, n * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_in, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    for (int bit = 0; bit < BITS; bit++) {
        preds_kernel<T><<<gridSize, blockSize>>>(d_in, d_preds, n, bit);
        
        exclusive_scan(d_scanned_preds, d_preds, n);
        
        int last_pred, last_scan;
        CUDA_CHECK(cudaMemcpy(&last_pred, &d_preds[n - 1], sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_scan, &d_scanned_preds[n - 1], sizeof(int), cudaMemcpyDeviceToHost));
        int total_zeros = last_scan + last_pred;
        
        scatter_kernel<T><<<gridSize, blockSize>>>(d_in, d_out, d_scanned_preds, n, bit, total_zeros);
        
        std::swap(d_in, d_out);
    }
    
    CUDA_CHECK(cudaMemcpy(h_data, d_in, n * sizeof(T), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_preds));
    CUDA_CHECK(cudaFree(d_scanned_preds));
}

template<typename T>
void test_radix_sort(int n) {
    std::vector<T> data(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (sizeof(T) == 4) {
        std::uniform_int_distribution<int32_t> dist(1, n);
        for (int i = 0; i < n; i++) {
            data[i] = static_cast<T>(dist(gen));
        }
    } else {
        std::uniform_int_distribution<int64_t> dist(1, n);
        for (int i = 0; i < n; i++) {
            data[i] = static_cast<T>(dist(gen));
        }
    }

    std::vector<T> cpu_data = data;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(cpu_data.begin(), cpu_data.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    std::vector<T> gpu_radix_data = data;
    
    cudaEvent_t start_radix, stop_radix;
    CUDA_CHECK(cudaEventCreate(&start_radix));
    CUDA_CHECK(cudaEventCreate(&stop_radix));
    
    CUDA_CHECK(cudaEventRecord(start_radix));
    radix_sort(gpu_radix_data.data(), n);
    CUDA_CHECK(cudaEventRecord(stop_radix));
    CUDA_CHECK(cudaEventSynchronize(stop_radix));
    
    float radix_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&radix_ms, start_radix, stop_radix));
    float radix_duration = radix_ms / 1000.0f;

    bool radix_sorted = true;
    for (int i = 1; i < n; i++) {
        if (gpu_radix_data[i] < gpu_radix_data[i-1]) {
            radix_sorted = false;
            break;
        }
    }

    std::vector<T> thrust_data = data;
    
    cudaEvent_t start_thrust, stop_thrust;
    CUDA_CHECK(cudaEventCreate(&start_thrust));
    CUDA_CHECK(cudaEventCreate(&stop_thrust));
    
    CUDA_CHECK(cudaEventRecord(start_thrust));
    
    thrust::device_vector<T> d_data(thrust_data.begin(), thrust_data.end());
    thrust::sort(d_data.begin(), d_data.end());
    thrust::copy(d_data.begin(), d_data.end(), thrust_data.begin());
    
    CUDA_CHECK(cudaEventRecord(stop_thrust));
    CUDA_CHECK(cudaEventSynchronize(stop_thrust));
    
    float thrust_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&thrust_ms, start_thrust, stop_thrust));
    float thrust_duration = thrust_ms / 1000.0f;

    bool thrust_sorted = true;
    for (int i = 1; i < n && i < 1000; i++) {
        if (thrust_data[i] < thrust_data[i-1]) {
            thrust_sorted = false;
            break;
        }
    }

    std::cout << "CPU sort time: " << cpu_duration.count() << " sec" << std::endl;
    std::cout << "GPU Radix Sort time: " << radix_duration << " sec" << std::endl;
    std::cout << "GPU thrust::sort time: " << thrust_duration << " sec" << std::endl;
    std::cout << "GPU Radix Sort is " << cpu_duration.count() / radix_duration << "x faster than CPU" << std::endl;
    std::cout << "GPU thrust::sort is " << cpu_duration.count() / thrust_duration << "x faster than CPU" << std::endl;
    std::cout << "Radix Sort: " << (radix_sorted ? "Success" : "Failed") << std::endl;
    std::cout << "Thrust Sort: " << (thrust_sorted ? "Success" : "Failed") << std::endl << std::endl;

    CUDA_CHECK(cudaEventDestroy(start_radix));
    CUDA_CHECK(cudaEventDestroy(stop_radix));
    CUDA_CHECK(cudaEventDestroy(start_thrust));
    CUDA_CHECK(cudaEventDestroy(stop_thrust));
}

int main() {
    std::cout << "100000 32-bit integers" << std::endl;
    test_radix_sort<int32_t>(100000);

    std::cout << "1000000 32-bit integers" << std::endl;
    test_radix_sort<int32_t>(1000000);

    std::cout << "100000000 32-bit integers" << std::endl;
    test_radix_sort<int32_t>(100000000);
    
    std::cout << "100000 64-bit integers" << std::endl;
    test_radix_sort<int64_t>(100000);

    std::cout << "1000000 64-bit integers" << std::endl;
    test_radix_sort<int64_t>(1000000);

    std::cout << "100000000 64-bit integers" << std::endl;
    test_radix_sort<int64_t>(100000000);
    
    return 0;
}
