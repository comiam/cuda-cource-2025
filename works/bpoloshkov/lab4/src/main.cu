#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void countDigits(unsigned int* input, unsigned int* counts, int n, int shift, int numBlocks) {
    __shared__ unsigned int warpCounts[8][RADIX_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    int laneId = tid & 31;
    int warpId = tid >> 5;
    int numWarps = BLOCK_SIZE >> 5;

    for (int i = tid; i < numWarps * RADIX_SIZE; i += BLOCK_SIZE) {
        ((unsigned int*)warpCounts)[i] = 0;
    }
    __syncthreads();

    unsigned int myDigit = RADIX_SIZE;
    if (gid < n) {
        myDigit = (input[gid] >> shift) & (RADIX_SIZE - 1);
    }

    unsigned int peersMask = 0;
    for (int i = 0; i < 32; i++) {
        unsigned int otherDigit = __shfl_sync(0xFFFFFFFF, myDigit, i);
        if (otherDigit == myDigit) peersMask |= (1u << i);
    }

    unsigned int countInWarp = __popc(peersMask);
    int firstLane = __ffs(peersMask) - 1;

    if (laneId == firstLane && myDigit < RADIX_SIZE) {
        warpCounts[warpId][myDigit] = countInWarp;
    }
    __syncthreads();

    if (tid < RADIX_SIZE) {
        unsigned int total = 0;
        for (int w = 0; w < numWarps; w++) {
            total += warpCounts[w][tid];
        }
        counts[tid * numBlocks + bid] = total;
    }
}

__global__ void prefixSumAndOffset(unsigned int* counts, int numBlocks, int n) {
    __shared__ unsigned int digitTotals[RADIX_SIZE];
    __shared__ unsigned int digitOffsets[RADIX_SIZE];
    
    int tid = threadIdx.x;
    
    if (tid < RADIX_SIZE) {
        unsigned int sum = 0;
        for (int b = 0; b < numBlocks; b++) {
            unsigned int val = counts[tid * numBlocks + b];
            counts[tid * numBlocks + b] = sum;
            sum += val;
        }
        digitTotals[tid] = sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        unsigned int offset = 0;
        for (int d = 0; d < RADIX_SIZE; d++) {
            digitOffsets[d] = offset;
            offset += digitTotals[d];
        }
    }
    __syncthreads();
    
    if (tid < RADIX_SIZE) {
        unsigned int globalOffset = digitOffsets[tid];
        for (int b = 0; b < numBlocks; b++) {
            counts[tid * numBlocks + b] += globalOffset;
        }
    }
}

__global__ void computeRanksAndScatter(unsigned int* input, unsigned int* output,
                                        unsigned int* blockOffsets, int n, int shift, int numBlocks) {
    __shared__ unsigned int digitBase[RADIX_SIZE];
    __shared__ unsigned int warpDigitCounts[8][RADIX_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    int laneId = tid & 31;
    int warpId = tid >> 5;
    int numWarps = BLOCK_SIZE >> 5;

    for (int i = tid; i < numWarps * RADIX_SIZE; i += BLOCK_SIZE) {
        ((unsigned int*)warpDigitCounts)[i] = 0;
    }

    if (tid < RADIX_SIZE) {
        digitBase[tid] = blockOffsets[tid * numBlocks + bid];
    }
    __syncthreads();

    bool valid = (gid < n);
    unsigned int myValue = 0;
    unsigned int myDigit = RADIX_SIZE;
    
    if (valid) {
        myValue = input[gid];
        myDigit = (myValue >> shift) & (RADIX_SIZE - 1);
    }

    unsigned int peersMask = 0;
    for (int i = 0; i < 32; i++) {
        unsigned int otherDigit = __shfl_sync(0xFFFFFFFF, myDigit, i);
        if (otherDigit == myDigit) peersMask |= (1u << i);
    }

    unsigned int rankInWarp = __popc(peersMask & ((1u << laneId) - 1));
    unsigned int countInWarp = __popc(peersMask);
    
    if (rankInWarp == countInWarp - 1 && myDigit < RADIX_SIZE) {
        warpDigitCounts[warpId][myDigit] = countInWarp;
    }
    __syncthreads();

    if (valid) {
        unsigned int rankPrevWarps = 0;
        for (int w = 0; w < warpId; w++) {
            rankPrevWarps += warpDigitCounts[w][myDigit];
        }
        unsigned int pos = digitBase[myDigit] + rankPrevWarps + rankInWarp;
        output[pos] = myValue;
    }
}

void radixSortUnsigned(unsigned int* d_input, unsigned int* d_output, int n) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, RADIX_SIZE * numBlocks * sizeof(unsigned int)));

    unsigned int* d_src = d_input;
    unsigned int* d_dst = d_output;

    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        CUDA_CHECK(cudaMemset(d_counts, 0, RADIX_SIZE * numBlocks * sizeof(unsigned int)));

        countDigits<<<numBlocks, BLOCK_SIZE>>>(d_src, d_counts, n, shift, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        prefixSumAndOffset<<<1, BLOCK_SIZE>>>(d_counts, numBlocks, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        computeRanksAndScatter<<<numBlocks, BLOCK_SIZE>>>(d_src, d_dst, d_counts, n, shift, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int* temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }

    if (d_src != d_input) {
        CUDA_CHECK(cudaMemcpy(d_input, d_src, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_counts);
}

__global__ void toUnsigned(int* input, unsigned int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (unsigned int)(input[idx] ^ 0x80000000);
    }
}

__global__ void toSigned(unsigned int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (int)(input[idx] ^ 0x80000000);
    }
}

void radixSortSigned(int* d_input, int* d_output, unsigned int* d_temp1, unsigned int* d_temp2, int n) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    toUnsigned<<<numBlocks, BLOCK_SIZE>>>(d_input, d_temp1, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    radixSortUnsigned(d_temp1, d_temp2, n);

    toSigned<<<numBlocks, BLOCK_SIZE>>>(d_temp1, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int comparator(const void* a, const void* b) {
    int va = *(int*)a;
    int vb = *(int*)b;
    return (va > vb) - (va < vb);
}

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool verifySorted(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int n = 1000000;

    if (argc > 1) {
        n = atoi(argv[1]);
        if (n < 1000 || n > 100000000) {
            fprintf(stderr, "Error: n must be between 1000 and 100000000\n");
            return 1;
        }
    }

    printf("Array size: %d\n\n", n);

    int* h_data = (int*)malloc(n * sizeof(int));
    int* h_copy = (int*)malloc(n * sizeof(int));
    int* h_result = (int*)malloc(n * sizeof(int));

    srand(42);
    for (int i = 0; i < n; i++) {
        h_data[i] = rand() - RAND_MAX / 2;
    }

    memcpy(h_copy, h_data, n * sizeof(int));
    double startCPU = getTime();
    if (n <= 10000000) {
        qsort(h_copy, n, sizeof(int), comparator);
    }
    double timeCPU = getTime() - startCPU;

    int *d_data, *d_output;
    unsigned int *d_temp1, *d_temp2;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp1, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_temp2, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    radixSortSigned(d_data, d_output, d_temp1, d_temp2, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeRadix;
    cudaEventElapsedTime(&timeRadix, start, stop);
    timeRadix /= 1000.0f;

    CUDA_CHECK(cudaMemcpy(h_result, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice));
    thrust::device_ptr<int> d_ptr(d_data);

    cudaEventRecord(start);
    thrust::sort(d_ptr, d_ptr + n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeThrust;
    cudaEventElapsedTime(&timeThrust, start, stop);
    timeThrust /= 1000.0f;

    if (n <= 10000000) {
        printf("CPU qsort time:     %.4f sec\n", timeCPU);
    } else {
        printf("CPU qsort time:     skipped (n > 10M)\n");
    }
    printf("GPU Radix Sort:     %.4f sec\n", timeRadix);
    printf("GPU thrust::sort:   %.4f sec\n", timeThrust);

    bool correct = verifySorted(h_result, n);
    printf("\nVerification: %s\n", correct ? "PASSED" : "FAILED");

    if (n <= 10000000 && timeCPU > 0) {
        printf("\nSpeedup vs CPU:     %.1fx\n", timeCPU / timeRadix);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    free(h_data);
    free(h_copy);
    free(h_result);

    return 0;
}
