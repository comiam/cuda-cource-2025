#include "radix_sort.h"
#include <stdio.h>
#include <cstring>

__global__ void scanUpSweep(uint32_t* data, int stride, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2 + stride - 1;
    if (idx + stride < size) {
        data[idx + stride] += data[idx]; // суммируем элементы
    }
}

__global__ void scanDownSweep(uint32_t* data, int stride, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2 + stride - 1;
    if (idx + stride < size) {
        uint32_t temp = data[idx]; // сохраняем старое значение
        data[idx] = data[idx + stride];
        data[idx + stride] += temp; // добавляем к следующему
    }
}

// простой скан для маленьких массивов (один блок)
__global__ void simpleScanKernel(uint32_t* data, int size) {
    __shared__ uint32_t temp[512];
    
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    
    if (tid < size) {
        temp[tid] = data[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    for (int offset = 1; offset < size; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pin;
        
        if (tid >= offset && tid < size) {
            temp[pout * 256 + tid] = temp[pin * 256 + tid] + temp[pin * 256 + tid - offset];
        } else if (tid < size) {
            temp[pout * 256 + tid] = temp[pin * 256 + tid];
        }
        __syncthreads();
    }
    
    if (tid < size) {
        if (tid == 0) {
            data[0] = 0;
        } else {
            data[tid] = temp[pout * 256 + tid - 1];
        }
    }
}

void prefixScan(uint32_t* d_data, int size) {
    // для маленьких размеров (RADIX=256) используем простой scan в одном блоке
    if (size <= 256) {
        simpleScanKernel<<<1, 256>>>(d_data, size);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    
    // для больших размеров - округляем до степени двойки
    int paddedSize = 1;
    while (paddedSize < size) paddedSize *= 2;
    
    uint32_t* d_padded = nullptr;
    if (paddedSize != size) {
        CUDA_CHECK(cudaMalloc(&d_padded, paddedSize * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_padded, d_data, size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_padded + size, 0, (paddedSize - size) * sizeof(uint32_t)));
    } else {
        d_padded = d_data;
    }
    
    // подъем вверх
    for (int stride = 1; stride < paddedSize; stride *= 2) {
        int numThreads = paddedSize / (stride * 2);
        if (numThreads > 0) {
            int blocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
            scanUpSweep<<<blocks, BLOCK_SIZE>>>(d_padded, stride, paddedSize);
        }
    }
    
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_padded + paddedSize - 1, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // спуск вниз
    for (int stride = paddedSize / 2; stride > 0; stride /= 2) {
        int numThreads = paddedSize / (stride * 2);
        if (numThreads > 0) {
            int blocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
            scanDownSweep<<<blocks, BLOCK_SIZE>>>(d_padded, stride, paddedSize);
        }
    }
    
    if (d_padded != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, d_padded, size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(d_padded));
    }
}

// строим гистограмму
__global__ void histogramKernel(const uint32_t* input, uint32_t* histogram, 
                                int size, int shift) {
    
    __shared__ uint32_t s_hist[RADIX + 1]; // +1 для избежания bank conflicts
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // обнуляем гистограмму
    if (tid < RADIX) {
        s_hist[tid] = 0;
    }
    __syncthreads();
    
    // считаем каждый элемент
    if (idx < size) {
        uint32_t value = input[idx];
        uint32_t digit = (value >> shift) & (RADIX - 1);
        atomicAdd(&s_hist[digit], 1);
    }
    __syncthreads();
    
    // записываем в глобальную память
    // для каждого блока сохраняем отдельную гистограмму: histogram[blockIdx * RADIX + digit]
    if (tid < RADIX) {
        histogram[blockIdx.x * RADIX + tid] = s_hist[tid];
    }
}

// суммируем гистограммы всех блоков
__global__ void sumHistogramsKernel(const uint32_t* blockHistograms, uint32_t* globalHistogram, int numBlocks) {
    int digit = threadIdx.x;
    if (digit < RADIX) {
        uint32_t sum = 0;
        for (int b = 0; b < numBlocks; b++) {
            sum += blockHistograms[b * RADIX + digit];
        }
        globalHistogram[digit] = sum;
    }
}

// вычисляем prefix scan для каждого блока на основе глобальных смещений
__global__ void computeBlockOffsetsKernel(const uint32_t* blockHistograms, const uint32_t* globalOffsets, 
                                         uint32_t* blockOffsets, int numBlocks) {
    int digit = threadIdx.x;
    int blockId = blockIdx.x;
    
    if (digit < RADIX) {
        uint32_t offset = globalOffsets[digit];
        // добавляем суммы из предыдущих блоков
        for (int b = 0; b < blockId; b++) {
            offset += blockHistograms[b * RADIX + digit];
        }
        blockOffsets[blockId * RADIX + digit] = offset;
    }
}

__global__ void reorderKernel(const uint32_t* input, uint32_t* output, 
                              const uint32_t* blockOffsets, int size, int shift) {
    
    __shared__ uint32_t s_baseOffsets[RADIX]; // базовые смещения для блока
    __shared__ uint32_t s_keys[BLOCK_SIZE]; // ключи
    __shared__ uint32_t s_digits[BLOCK_SIZE]; // разряды
    __shared__ uint32_t s_localOffsets[BLOCK_SIZE]; // локальные смещения для каждого потока
    __shared__ uint32_t s_digitCounts[RADIX]; // счетчики для каждого разряда
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < RADIX) {
        s_baseOffsets[tid] = blockOffsets[blockIdx.x * RADIX + tid];
        s_digitCounts[tid] = 0;
    }
    __syncthreads();
    
    if (idx < size) {
        uint32_t key = input[idx];
        uint32_t digit = (key >> shift) & (RADIX - 1);
        
        s_keys[tid] = key;
        s_digits[tid] = digit;
    } else {
        s_digits[tid] = 0;
    }
    __syncthreads();
    
    if (idx < size) {
        uint32_t myDigit = s_digits[tid];
        
        uint32_t localOffset = 0;
        
        // делим на 32 элемента (типа как варп)
        for (int base = 0; base < BLOCK_SIZE; base += 32) {
            int limit = min(base + 32, tid);
            for (int i = base; i < limit; i++) {
                if (s_digits[i] == myDigit) {
                    localOffset++;
                }
            }
        }
        
        s_localOffsets[tid] = localOffset;
        atomicMax(&s_digitCounts[myDigit], localOffset + 1);
    }
    __syncthreads();
    
    if (idx < size) {
        uint32_t digit = s_digits[tid];
        uint32_t globalPos = s_baseOffsets[digit] + s_localOffsets[tid];
        output[globalPos] = s_keys[tid];
    }
}

// radix sort для беззнаковых 32-бит чисел
void radixSortUInt32(uint32_t* d_data, int size) {
    uint32_t* d_temp; 
    uint32_t* d_blockHistograms;
    uint32_t* d_globalHistogram;
    uint32_t* d_blockOffsets;
    
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_blockHistograms, numBlocks * RADIX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_globalHistogram, RADIX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, numBlocks * RADIX * sizeof(uint32_t)));
    
    uint32_t* current = d_data;
    uint32_t* next = d_temp;
    
    // обрабатываем по 8 бит за раз
    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(current, d_blockHistograms, size, shift);
        CUDA_CHECK(cudaGetLastError());
        
        sumHistogramsKernel<<<1, RADIX>>>(d_blockHistograms, d_globalHistogram, numBlocks);
        CUDA_CHECK(cudaGetLastError());
        

        prefixScan(d_globalHistogram, RADIX);
        
        computeBlockOffsetsKernel<<<numBlocks, RADIX>>>(d_blockHistograms, d_globalHistogram, d_blockOffsets, numBlocks);
        CUDA_CHECK(cudaGetLastError());
        
        reorderKernel<<<numBlocks, BLOCK_SIZE>>>(current, next, d_blockOffsets, size, shift);
        CUDA_CHECK(cudaGetLastError());
        
        uint32_t* temp = current;
        current = next;
        next = temp;
    }
    
    if (current != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, current, size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    }
    
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_blockHistograms));
    CUDA_CHECK(cudaFree(d_globalHistogram));
    CUDA_CHECK(cudaFree(d_blockOffsets));
}

// kernel для инверсии знакового бита (32-бит)
__global__ void flipSignBit32Kernel(uint32_t* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] ^= 0x80000000;
    }
}

// radix sort для знаковых 32-бит чисел
void radixSortInt32(int* d_data, int size) {
    uint32_t* d_unsigned = reinterpret_cast<uint32_t*>(d_data);
    
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    flipSignBit32Kernel<<<numBlocks, BLOCK_SIZE>>>(d_unsigned, size); // инвертируем перед сортировкой
    CUDA_CHECK(cudaGetLastError());
    
    radixSortUInt32(d_unsigned, size); // сортируем как беззнаковые
    
    flipSignBit32Kernel<<<numBlocks, BLOCK_SIZE>>>(d_unsigned, size); // возвращаем знаковый бит
    CUDA_CHECK(cudaGetLastError());
}

// гистограмма для 64-бит чисел
__global__ void histogram64Kernel(const uint64_t* input, uint32_t* histogram, 
                                  int size, int shift) {
    
    __shared__ uint32_t s_hist[RADIX + 1]; // +1 для избежания bank conflicts
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // обнуляем
    if (tid < RADIX) {
        s_hist[tid] = 0;
    }
    __syncthreads();
    
    // считаем
    if (idx < size) {
        uint64_t value = input[idx];
        uint32_t digit = (value >> shift) & (RADIX - 1);
        atomicAdd(&s_hist[digit], 1);
    }
    __syncthreads();
    
    if (tid < RADIX) {
        histogram[blockIdx.x * RADIX + tid] = s_hist[tid];
    }
}

// перестановка для 64-бит чисел
__global__ void reorder64Kernel(const uint64_t* input, uint64_t* output, 
                                const uint32_t* blockOffsets, int size, int shift) {
    __shared__ uint32_t s_baseOffsets[RADIX];
    __shared__ uint64_t s_keys[BLOCK_SIZE]; 
    __shared__ uint32_t s_digits[BLOCK_SIZE];
    __shared__ uint32_t s_localOffsets[BLOCK_SIZE];
    __shared__ uint32_t s_digitCounts[RADIX];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < RADIX) {
        s_baseOffsets[tid] = blockOffsets[blockIdx.x * RADIX + tid];
        s_digitCounts[tid] = 0;
    }
    __syncthreads();
    
    if (idx < size) {
        uint64_t key = input[idx];
        uint32_t digit = (key >> shift) & (RADIX - 1);
        
        s_keys[tid] = key;
        s_digits[tid] = digit;
    } else {
        s_digits[tid] = 0;
    }
    __syncthreads();
    
    if (idx < size) {
        uint32_t myDigit = s_digits[tid];
        uint32_t localOffset = 0;
        
        // оптимизация: делим работу на проходы по 32 элемента (warp)
        for (int base = 0; base < BLOCK_SIZE; base += 32) {
            int limit = min(base + 32, tid);
            for (int i = base; i < limit; i++) {
                if (s_digits[i] == myDigit) {
                    localOffset++;
                }
            }
        }
        
        s_localOffsets[tid] = localOffset;
        atomicMax(&s_digitCounts[myDigit], localOffset + 1);
    }
    __syncthreads();
    
    if (idx < size) {
        uint32_t digit = s_digits[tid];
        uint32_t globalPos = s_baseOffsets[digit] + s_localOffsets[tid];
        output[globalPos] = s_keys[tid];
    }
}

// radix sort для беззнаковых 64-бит чисел
void radixSortUInt64(uint64_t* d_data, int size) {
    uint64_t* d_temp; 
    uint32_t* d_blockHistograms;
    uint32_t* d_globalHistogram;
    uint32_t* d_blockOffsets;
    
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    CUDA_CHECK(cudaMalloc(&d_temp, size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_blockHistograms, numBlocks * RADIX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_globalHistogram, RADIX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, numBlocks * RADIX * sizeof(uint32_t)));
    
    uint64_t* current = d_data;
    uint64_t* next = d_temp;
    
    // 8 проходов по 8 бит
    for (int shift = 0; shift < 64; shift += RADIX_BITS) {
        histogram64Kernel<<<numBlocks, BLOCK_SIZE>>>(current, d_blockHistograms, size, shift);
        CUDA_CHECK(cudaGetLastError());
        
        sumHistogramsKernel<<<1, RADIX>>>(d_blockHistograms, d_globalHistogram, numBlocks);
        CUDA_CHECK(cudaGetLastError());
        
        prefixScan(d_globalHistogram, RADIX);
        
        computeBlockOffsetsKernel<<<numBlocks, RADIX>>>(d_blockHistograms, d_globalHistogram, d_blockOffsets, numBlocks);
        CUDA_CHECK(cudaGetLastError());
        
        reorder64Kernel<<<numBlocks, BLOCK_SIZE>>>(current, next, d_blockOffsets, size, shift);
        CUDA_CHECK(cudaGetLastError());
        
        uint64_t* temp = current;
        current = next;
        next = temp;
    }
    
    if (current != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, current, size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    }
    
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_blockHistograms));
    CUDA_CHECK(cudaFree(d_globalHistogram));
    CUDA_CHECK(cudaFree(d_blockOffsets));
}

// kernel для инверсии знакового бита (64-бит)
__global__ void flipSignBit64Kernel(uint64_t* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] ^= 0x8000000000000000ULL;
    }
}

// radix sort для знаковых 64-бит чисел
void radixSortInt64(int64_t* d_data, int size) {
    uint64_t* d_unsigned = reinterpret_cast<uint64_t*>(d_data);
    
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    flipSignBit64Kernel<<<numBlocks, BLOCK_SIZE>>>(d_unsigned, size);
    CUDA_CHECK(cudaGetLastError());
    
    radixSortUInt64(d_unsigned, size);
    
    flipSignBit64Kernel<<<numBlocks, BLOCK_SIZE>>>(d_unsigned, size);
    CUDA_CHECK(cudaGetLastError());
}

// проверка что массив отсортирован
bool verifySorted(const int* data, int size) {
    for (int i = 1; i < size; i++) {
        if (data[i] < data[i-1]) { 
            printf("Error at index %d: %d > %d\n", i-1, data[i-1], data[i]);
            return false;
        }
    }
    return true; 
}

// вывод массива
void printArray(const int* data, int size, int maxPrint) {
    int n = (size < maxPrint) ? size : maxPrint;
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d", data[i]);
        if (i < n - 1) printf(", ");
    }
    if (size > maxPrint) printf(", ...");
    printf("]\n");
}
