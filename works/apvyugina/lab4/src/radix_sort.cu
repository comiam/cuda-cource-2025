#include <stdio.h>
#include <vector>
#include <chrono>
#include <stdint.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (2*THREADS_PER_BLOCK)
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) (((n) >> LOG_NUM_BANKS))


// Multi-block scan: Phase 1 - Local exclusive scan in each block
__global__
void scanSingleBlock(uint32_t* input, uint32_t* output, uint32_t* blockSums, int totalN) {
    extern __shared__ uint32_t temp[];
    
    int n = ELEMENTS_PER_BLOCK;
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    
    int blockOffset = blockId * n;
    
    // Load data into shared memory
    int ai = threadId;
    int bi = threadId + (n / 2);
    
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    temp[ai + bankOffsetA] = (blockOffset + ai < totalN) ? input[blockOffset + ai] : 0;
    temp[bi + bankOffsetB] = (blockOffset + bi < totalN) ? input[blockOffset + bi] : 0;
    

    int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1){ // build sum in place up the tree
		__syncthreads();
		if (threadId < d)
		{
			int ai = offset * (2 * threadId + 1) - 1;
			int bi = offset * (2 * threadId + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();

    if (threadId == 0) {
        blockSums[blockId] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

	for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (threadId < d){
			int ai = offset * (2 * threadId + 1) - 1;
			int bi = offset * (2 * threadId + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
    
    if (blockOffset + ai < totalN) {
        output[blockOffset + ai] = temp[ai + bankOffsetA];
    }
    if (blockOffset + bi < totalN) {
        output[blockOffset + bi] = temp[bi + bankOffsetB];
    }
}


// Multi-block scan: Phase 2 - Add block prefix sums
__global__
void addMultiBlockSums(uint32_t* output, uint32_t* blockPrefixSums, int n, int elementsPerBlock) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
    int blockOffset = blockId * ELEMENTS_PER_BLOCK;
    int idx = blockOffset + threadId;
    if (idx < n) {
        output[idx] += blockPrefixSums[blockId];
    }
}


template<typename T>
__global__
void extractBitsKernel(T* input, uint32_t* bits, int n, unsigned int bitPos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bits[idx] = (input[idx] >> bitPos) & 1u;
    }
}

template<typename T>
__global__
void scatterKernel(T* input, T* output, uint32_t* scanResult, uint32_t* bits, int n, int numZeros) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bit = (int)bits[idx];
        int pos;
        if (bit == 0) {
            // scanResult[idx] = number of 1s before position idx
            // Number of 0s before idx = idx - scanResult[idx]
            pos = idx - (int)scanResult[idx];
        } else {
            // For 1s: position = numZeros + number of 1s before this position
            pos = numZeros + (int)scanResult[idx];
        }
        if (pos >= 0 && pos < n) {
            output[pos] = input[idx];
        }
    }
}


void scanKernel(uint32_t* input, uint32_t* output, int n) {
    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    
    uint32_t* d_blockSums;
    uint32_t* d_blockPrefixSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(uint32_t));
    cudaMalloc(&d_blockPrefixSums, numBlocks * sizeof(uint32_t));

    // Step 1: Local scan in each block
    int blockSharedMem = (ELEMENTS_PER_BLOCK + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK)) * sizeof(uint32_t);
    scanSingleBlock<<<numBlocks, (ELEMENTS_PER_BLOCK + 1) / 2, blockSharedMem>>>(
        input, output, d_blockSums, n
    );
    cudaDeviceSynchronize();
    
    // Step 2: Scan block sums (recursive call)
    if (numBlocks > 1) {
        scanKernel(d_blockSums, d_blockPrefixSums, numBlocks);
    } else {
        // Single block case: prefix sum is 0
        cudaMemset(d_blockPrefixSums, 0, sizeof(uint32_t));
    }
    cudaDeviceSynchronize();
    
    // Step 3: Add block prefix sums to each block
    addMultiBlockSums<<<numBlocks, ELEMENTS_PER_BLOCK>>>(output, d_blockPrefixSums, n, ELEMENTS_PER_BLOCK);
    cudaDeviceSynchronize();
    
    cudaFree(d_blockSums);
    cudaFree(d_blockPrefixSums);
}


template<typename T>
void radixSortTemplate(T* d_input, T* d_output, int n) {
    T* d_temp1 = d_input;
    T* d_temp2 = d_output;
    
    // Determine number of bits based on type size
    const int numBits = sizeof(T) * 8;
    
    uint32_t* d_bits;
    uint32_t* d_scanResult;
    cudaMalloc(&d_bits, n * sizeof(uint32_t));
    cudaMalloc(&d_scanResult, n * sizeof(uint32_t));
    
    int blocksPerGrid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Process each bit from LSB to MSB
    for (int bit = 0; bit < numBits; bit++) {
        // Extract bits
        extractBitsKernel<T><<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_temp1, d_bits, n, bit);
        cudaDeviceSynchronize();

        scanKernel(d_bits, d_scanResult, n);
        cudaDeviceSynchronize(); 

        // Count total zeros (only from first n elements)
        // scanResult[i] = exclusive prefix sum = sum of bits[0..i-1]
        // Total sum of all bits = scanResult[n-1] + bits[n-1]
        // numZeros = n - total sum
        uint32_t lastScan, lastBit;
        cudaMemcpy(&lastScan, &d_scanResult[n-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastBit, &d_bits[n-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
        int numZeros = n - (int)(lastScan + lastBit);
        
        // Scatter elements
        scatterKernel<T><<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_temp1, d_temp2, d_scanResult, d_bits, n, numZeros);
        cudaDeviceSynchronize();
        
        // Swap buffers
        T* temp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp;
    }
    
    // Copy final result to output if needed
    if (d_temp1 != d_output) {
        cudaMemcpy(d_output, d_temp1, n * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_bits);
    cudaFree(d_scanResult);
}

void radixSort_int32(uint32_t* d_input, uint32_t* d_output, int n){
    radixSortTemplate<uint32_t>(d_input, d_output, n);
}

void radixSort_int64(uint64_t* d_input, uint64_t* d_output, int n){
    radixSortTemplate<uint64_t>(d_input, d_output, n);
}