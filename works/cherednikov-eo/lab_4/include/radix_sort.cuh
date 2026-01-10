//
// Created by Egor on 09.01.2026.
//

#ifndef CUDA_COURCE_2025_RADIX_SORT_CUH
#define CUDA_COURCE_2025_RADIX_SORT_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Radix Sort functions for different integer types
// All functions take input and output device pointers and array size
// Input and output can be the same pointer (in-place sorting)

// Sort 32-bit unsigned integers
void radixSort_uint32(uint32_t* d_input, uint32_t* d_output, int n);

// Sort 64-bit unsigned integers
void radixSort_uint64(uint64_t* d_input, uint64_t* d_output, int n);

// Sort 32-bit signed integers (handles negative numbers)
void radixSort_int32(int32_t* d_input, int32_t* d_output, int n);

// Sort 64-bit signed integers (handles negative numbers)
void radixSort_int64(int64_t* d_input, int64_t* d_output, int n);

#endif //CUDA_COURCE_2025_RADIX_SORT_CUH