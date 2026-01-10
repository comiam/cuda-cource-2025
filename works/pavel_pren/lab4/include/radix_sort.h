#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include <cuda_runtime.h>
#include <cstdint>

// функция проверки ошибок (из лекции)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

const int BLOCK_SIZE = 256; // размер блока потоков
const int RADIX_BITS = 8; // сколько бит обрабатываем за раз
const int RADIX = (1 << RADIX_BITS); // количество разрядов (256)

void radixSortInt32(int* d_data, int size); 
void radixSortInt64(int64_t* d_data, int size); 
void radixSortUInt32(uint32_t* d_data, int size); 
void radixSortUInt64(uint64_t* d_data, int size);

__global__ void histogramKernel(const uint32_t* input, uint32_t* histogram,
                                int size, int shift);
__global__ void histogram64Kernel(const uint64_t* input, uint32_t* histogram,
                                  int size, int shift);
__global__ void scanKernel(uint32_t* data, uint32_t* sums, int size);
__global__ void scanAddKernel(uint32_t* data, const uint32_t* sums, int size);
__global__ void reorderKernel(const uint32_t* input, uint32_t* output,
                              const uint32_t* histogram, int size, int shift);
__global__ void reorder64Kernel(const uint64_t* input, uint64_t* output,
                                const uint32_t* blockOffsets, int size, int shift);

void prefixScan(uint32_t* d_data, int size);
bool verifySorted(const int* data, int size);
void printArray(const int* data, int size, int maxPrint = 20);

#endif
