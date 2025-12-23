#include "../include/radix_sort.h"
#include "../include/error_check.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>

__global__ void prescan_kernel(const unsigned int* input, unsigned int* output, unsigned int* block_sums, int n) {
    extern __shared__ unsigned int temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        unsigned int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    if (block_sums != nullptr && tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = temp[tid];
    }

    unsigned int res = (tid > 0) ? temp[tid - 1] : 0;
    if (idx < n) {
        output[idx] = res;
    }
}

__global__ void add_block_sums_kernel(unsigned int* output, const unsigned int* scanned_block_sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] += scanned_block_sums[blockIdx.x];
    }
}

void scan_recursive(unsigned int* input, unsigned int* output, int n, unsigned int* aux_buffer) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    unsigned int* sums = aux_buffer;
    unsigned int* next_aux = aux_buffer + blocks;

    prescan_kernel<<<blocks, threads, threads * sizeof(unsigned int)>>>(input, output, sums, n);
    CUDA_CHECK(cudaGetLastError());

    if (blocks > 1) {
        scan_recursive(sums, sums, blocks, next_aux);
        add_block_sums_kernel<<<blocks, threads>>>(output, sums, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
__global__ void toggle_sign_bit_kernel(T* data, int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        constexpr T sign_mask = T(1) << (sizeof(T) * 8 - 1);
        data[index] ^= sign_mask;
    }
}

template <typename T>
__global__ void predicate_kernel(const T* input, unsigned int* flags, int bit, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        T val = input[index];
        int bit_val = (val >> bit) & 1;
        flags[index] = (bit_val == 0) ? 1 : 0;
    }
}

template <typename T>
__global__ void scatter_kernel(const T* input, T* output, 
                               const unsigned int* flags, const unsigned int* scanned_flags, 
                               int bit, int n, int total_zeros) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        T val = input[index];
        
        unsigned int is_zero_group = flags[index];
        unsigned int prefix_zeros = scanned_flags[index];
        
        int dest_idx;
        if (is_zero_group) {
            dest_idx = prefix_zeros;
        } else {
            int prefix_ones = index - prefix_zeros;
            dest_idx = total_zeros + prefix_ones;
        }
        
        output[dest_idx] = val;
    }
}

template <typename T>
void radix_sort_gpu(T* device_input, T* device_temp, int num_elements) {
    bool is_signed = std::is_signed<T>::value;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    if (is_signed) {
        toggle_sign_bit_kernel<<<blocks, threads>>>(device_input, num_elements);
        CUDA_CHECK(cudaGetLastError());
    }

    unsigned int *device_flags, *device_scanned_flags, *device_scan_aux;
    CUDA_CHECK(cudaMalloc(&device_flags, num_elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&device_scanned_flags, num_elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&device_scan_aux, num_elements * sizeof(unsigned int)));

    T* device_source = device_input;
    T* device_destination = device_temp;

    const static int BYTE_SIZE = 8;

    int num_bits = sizeof(T) * BYTE_SIZE;

    for (int bit = 0; bit < num_bits; ++bit) {
        predicate_kernel<<<blocks, threads>>>(device_source, device_flags, bit, num_elements);
        CUDA_CHECK(cudaGetLastError());
        
        scan_recursive(device_flags, device_scanned_flags, num_elements, device_scan_aux);
        
        unsigned int last_flag, last_scanned;
        CUDA_CHECK(cudaMemcpy(&last_flag, &device_flags[num_elements - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_scanned, &device_scanned_flags[num_elements - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
        int total_zeros = last_scanned + last_flag;

        scatter_kernel<<<blocks, threads>>>(device_source, device_destination, device_flags, device_scanned_flags, bit, num_elements, total_zeros);
        CUDA_CHECK(cudaGetLastError());

        std::swap(device_source, device_destination);
    }
    
    if (device_source != device_input) {
        CUDA_CHECK(cudaMemcpy(device_input, device_source, num_elements * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(device_flags));
    CUDA_CHECK(cudaFree(device_scanned_flags));
    CUDA_CHECK(cudaFree(device_scan_aux));

    if (is_signed) {
        toggle_sign_bit_kernel<<<blocks, threads>>>(device_input, num_elements);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
void radix_sort_cpu(std::vector<T>& data) {
    std::sort(data.begin(), data.end());
}

template void radix_sort_gpu<int>(int*, int*, int);
template void radix_sort_gpu<long long>(long long*, long long*, int);
template void radix_sort_cpu<int>(std::vector<int>&);
template void radix_sort_cpu<long long>(std::vector<long long>&);
