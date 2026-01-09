#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <cmath>

#define BLOCK_SIZE 1024 
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define DIGITS_COUNT 256
#define RADIX_MASK 255

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
__device__ __forceinline__ unsigned int get_digit(T x, unsigned int shift) {
    return (unsigned int)((x >> shift) & RADIX_MASK);
}

__global__ void FlipSignBit(int* arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        ((unsigned int*)arr)[idx] ^= 0x80000000;
    }
}

__global__ void RadixSortCalcDigitsPerBlocks(
    const int* A,
    unsigned int* digitsPerBlock,
    unsigned int shift,
    unsigned int N,
    unsigned int BLOCKS_COUNT
) {
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    __shared__ unsigned int tile[DIGITS_COUNT];
    if(threadIdx.x < DIGITS_COUNT) tile[threadIdx.x] = 0;
    __syncthreads();

    if(col < N) {
        unsigned int digit = get_digit(((const unsigned int*)A)[col], shift);
        atomicAdd(&tile[digit], 1);
    }
    __syncthreads();

    if(threadIdx.x < DIGITS_COUNT) {
        digitsPerBlock[threadIdx.x * BLOCKS_COUNT + blockIdx.x] = tile[threadIdx.x];
    }
} 

__global__ void RadixSortScanHistogramBlelloch(
    unsigned int* digitsPerBlock,
    unsigned int* totalCountPerDigit,
    unsigned int BLOCKS_COUNT
) {
    extern __shared__ unsigned int temp[]; 
    int tid = threadIdx.x;
    int digit_row = blockIdx.x;
    unsigned int offset_idx = digit_row * BLOCKS_COUNT;
    int ai = tid;
    int bi = tid + blockDim.x;
    
    temp[ai] = (ai < BLOCKS_COUNT) ? digitsPerBlock[offset_idx + ai] : 0;
    temp[bi] = (bi < BLOCKS_COUNT) ? digitsPerBlock[offset_idx + bi] : 0;

    int n = 2 * blockDim.x; 
    int offset = 1;
    
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    
    if (tid == 0) {
        totalCountPerDigit[digit_row] = temp[n - 1];
        temp[n - 1] = 0;
    }
    
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    if (ai < BLOCKS_COUNT) digitsPerBlock[offset_idx + ai] = temp[ai];
    if (bi < BLOCKS_COUNT) digitsPerBlock[offset_idx + bi] = temp[bi];
}

__global__ void RadixSortScanHistogramHillisSteele(
    unsigned int* digitsPerBlock,
    unsigned int* totalCountPerDigit,
    unsigned int BLOCKS_COUNT
) {
    extern __shared__ unsigned int temp[]; 
    unsigned int* din = temp;
    unsigned int* dout = &temp[BLOCKS_COUNT];

    int tid = threadIdx.x;
    int digit_row = blockIdx.x;
    unsigned int offset_idx = digit_row * BLOCKS_COUNT;

    if (tid < BLOCKS_COUNT) {
        din[tid] = digitsPerBlock[offset_idx + tid];
    } else {
        if(tid < 2 * BLOCKS_COUNT) din[tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < BLOCKS_COUNT; offset *= 2) {
        if (tid < BLOCKS_COUNT) {
            dout[tid] = (tid >= offset)? din[tid] + din[tid - offset] : din[tid];
        }
        __syncthreads();
        
        unsigned int* swap = din; din = dout; dout = swap;
    }

    if (tid < BLOCKS_COUNT) {
        unsigned int val = din[tid];
        
        if (tid == BLOCKS_COUNT - 1) {
            totalCountPerDigit[digit_row] = val;
        }

        unsigned int exclusive_val = (tid == 0) ? 0 : din[tid - 1];
        digitsPerBlock[offset_idx + tid] = exclusive_val;
    }
}

__global__ void RadixSortScanBuckets(unsigned int* totalCountPerDigit) {
    __shared__ unsigned int temp[DIGITS_COUNT];
    int tid = threadIdx.x;
    int n = DIGITS_COUNT;
    int ai = tid; int bi = tid + (n / 2);

    temp[ai] = totalCountPerDigit[ai];
    temp[bi] = totalCountPerDigit[bi];
    int offset = 1;

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (tid == 0) temp[n - 1] = 0; 

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    totalCountPerDigit[ai] = temp[ai];
    totalCountPerDigit[bi] = temp[bi];
}

__global__ void RadixSortScatter(
    const int* A,
    int* R,
    const unsigned int* digitsPerBlock,
    const unsigned int* bucketOffsets,
    unsigned int shift,
    unsigned int N,
    unsigned int BLOCKS_COUNT
) {
    __shared__ unsigned int s_warp_counts[WARPS_PER_BLOCK][DIGITS_COUNT];
    __shared__ unsigned int s_global_bases[DIGITS_COUNT];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    for (int i = tid; i < WARPS_PER_BLOCK * DIGITS_COUNT; i += BLOCK_SIZE) {
        ((unsigned int*)s_warp_counts)[i] = 0;
    }

    if (tid < DIGITS_COUNT) {
        unsigned int global_bucket = bucketOffsets[tid];
        unsigned int block_off = digitsPerBlock[tid * BLOCKS_COUNT + blockIdx.x];
        s_global_bases[tid] = global_bucket + block_off;
    }
    __syncthreads();

    int col = blockIdx.x * BLOCK_SIZE + tid;
    unsigned int val = 0;
    unsigned int digit = 0;
    bool active = (col < N);

    unsigned int rank_in_warp = 0;
    unsigned int count_in_warp = 0;

    if (active) {
        val = ((const unsigned int*)A)[col];
        digit = get_digit(val, shift);

        unsigned int peers_bits = 0;
        #pragma unroll
        for(int i = 0; i < WARP_SIZE; i++) {
            unsigned int other_digit = __shfl_sync(0xFFFFFFFF, digit, i);
            if (other_digit == digit) peers_bits |= (1u << i);
        }
        rank_in_warp = __popc(peers_bits & ((1u << lane) - 1));
        count_in_warp = __popc(peers_bits);

        if (rank_in_warp == count_in_warp - 1) {
            s_warp_counts[warp_id][digit] = count_in_warp;
        }
    }
    __syncthreads();

    if (active) {
        unsigned int rank_prev_warps = 0;
        #pragma unroll
        for (int w = 0; w < warp_id; ++w) {
            rank_prev_warps += s_warp_counts[w][digit];
        }
        unsigned int global_base = s_global_bases[digit];
        unsigned int final_pos = global_base + rank_prev_warps + rank_in_warp;
        ((unsigned int*)R)[final_pos] = val;
    }
}

__global__ void BitonicSortStep(int *dev_values, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    if (ixj > i) {
        int val_i = dev_values[i];
        int val_ixj = dev_values[ixj];
        
        bool ascending = ((i & k) == 0);
        
        if ((ascending && (val_i > val_ixj)) || (!ascending && (val_i < val_ixj))) {
            dev_values[i] = val_ixj;
            dev_values[ixj] = val_i;
        }
    }
}

void RunRadixSort(int* d_A, int* d_R, int N, const char* scan_mode) {
    unsigned int BLOCKS_COUNT = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    unsigned int *d_digitsPerBlock, *d_totalCountPerDigit;
    size_t sizeDigits = BLOCKS_COUNT * DIGITS_COUNT * sizeof(unsigned int);
    size_t sizeTotal = DIGITS_COUNT * sizeof(unsigned int);
    
    gpuErrchk(cudaMalloc(&d_digitsPerBlock, sizeDigits));
    gpuErrchk(cudaMalloc(&d_totalCountPerDigit, sizeTotal));

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(BLOCKS_COUNT);
    dim3 dimPrefix(DIGITS_COUNT / 2);
    
    int scan_threads = 512;
    if (BLOCKS_COUNT > 1024) scan_threads = 1024;
    if (BLOCKS_COUNT > 2048 && std::string(scan_mode) == "Blelloch") {
        printf("Warning: Blelloch не умеет обрабатывать > 2048 блоков\n");
    }

    dim3 dimScanBlock(scan_threads);
    dim3 dimScanGrid(DIGITS_COUNT);

    FlipSignBit<<<dimGrid, dimBlock>>>(d_A, N);

    for(unsigned int shift = 0; shift < 32; shift += 8) {
        RadixSortCalcDigitsPerBlocks<<<dimGrid, dimBlock>>>(d_A, d_digitsPerBlock, shift, N, BLOCKS_COUNT);
        
        if (std::string(scan_mode) == "Blelloch") {
            size_t smem = 2 * scan_threads * sizeof(unsigned int);
            RadixSortScanHistogramBlelloch<<<dimScanGrid, dimScanBlock, smem>>>(d_digitsPerBlock, d_totalCountPerDigit, BLOCKS_COUNT);
        } else {
            int hs_threads = BLOCKS_COUNT;
            if (hs_threads > 1024) hs_threads = 1024; // костылёк, но для N=10^6 годится
            size_t smem = 2 * BLOCKS_COUNT * sizeof(unsigned int);
            RadixSortScanHistogramHillisSteele<<<dimScanGrid, hs_threads, smem>>>(d_digitsPerBlock, d_totalCountPerDigit, BLOCKS_COUNT);
        }
        
        RadixSortScanBuckets<<<1, dimPrefix>>>(d_totalCountPerDigit);
        
        RadixSortScatter<<<dimGrid, dimBlock>>>(d_A, d_R, d_digitsPerBlock, d_totalCountPerDigit, shift, N, BLOCKS_COUNT);
        
        std::swap(d_A, d_R);
    }

    FlipSignBit<<<dimGrid, dimBlock>>>(d_A, N);

    cudaFree(d_digitsPerBlock);
    cudaFree(d_totalCountPerDigit);
}

void RunBitonicSortWrapper(int* d_A, int padded_N) {
    int threads = 512;
    int blocks = padded_N / threads;

    for (int k = 2; k <= padded_N; k <<= 1) {       
        for (int j = k >> 1; j > 0; j >>= 1) { 
            BitonicSortStep<<<blocks, threads>>>(d_A, j, k);
        }
    }
}

int main() {
    int N_input;
    std::cout << "N:";
    if (!(std::cin >> N_input)) return 0;
    
    int N_padded = 1;
    while (N_padded < N_input) N_padded <<= 1;

    std::cout << "Data Size: " << N_input << "\n";
    if (N_padded != N_input) std::cout << "Padded Size (for Bitonic): " << N_padded << "\n";
    std::cout << "\n";
    
    size_t sizeInput = N_input * sizeof(int);
    size_t sizePadded = N_padded * sizeof(int);

    std::vector<int> h_A(N_input);
    std::vector<int> h_A_Padded(N_padded);
    
    srand(time(0));
    for(int i=0; i<N_input; ++i) {
        int r = (rand() << 16) | rand(); 
        if(rand()%2) r = -r;
        h_A[i] = r;
        h_A_Padded[i] = r;
    }
    
    for(int i=N_input; i<N_padded; ++i) {
        h_A_Padded[i] = std::numeric_limits<int>::max();
    }
    
    int *d_A, *d_R, *d_A_Bitonic;
    gpuErrchk(cudaMalloc(&d_A, sizeInput));
    gpuErrchk(cudaMalloc(&d_R, sizeInput));
    gpuErrchk(cudaMalloc(&d_A_Bitonic, sizePadded));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float elapsed;
    
    std::cout << "1. Run CPU std::sort...\n";
    std::vector<int> h_CPU = h_A;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::sort(h_CPU.begin(), h_CPU.end());
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();
    std::cout << "   Time: " << std::fixed << std::setprecision(4) << cpu_time << " s\n\n";
    
    std::cout << "2. Run GPU Radix Sort (Blelloch Scan)...\n";
    gpuErrchk(cudaMemcpy(d_A, h_A.data(), sizeInput, cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    RunRadixSort(d_A, d_R, N_input, "Blelloch");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "   Time: " << elapsed / 1000.0f << " s\n";
    
    std::vector<int> h_res(N_input);
    gpuErrchk(cudaMemcpy(h_res.data(), d_A, sizeInput, cudaMemcpyDeviceToHost));
    if (h_res == h_CPU) std::cout << "   Status: Correct\n\n";
    else std::cout << "   Status: INCORRECT!\n\n";

    std::cout << "3. Run GPU Radix Sort (Hillis-Steele Scan)...\n";
    gpuErrchk(cudaMemcpy(d_A, h_A.data(), sizeInput, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    RunRadixSort(d_A, d_R, N_input, "Hillis");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "   Time: " << elapsed / 1000.0f << " s\n";

    gpuErrchk(cudaMemcpy(h_res.data(), d_A, sizeInput, cudaMemcpyDeviceToHost));
    if (h_res == h_CPU) std::cout << "   Status: Correct\n\n";
    else std::cout << "   Status: INCORRECT!\n\n";

    std::cout << "4. Run GPU Bitonic Sort (Comparison Based, Padded input)...\n";
    gpuErrchk(cudaMemcpy(d_A_Bitonic, h_A_Padded.data(), sizePadded, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    RunBitonicSortWrapper(d_A_Bitonic, N_padded);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "   Time: " << elapsed / 1000.0f << " s\n";

    gpuErrchk(cudaMemcpy(h_res.data(), d_A_Bitonic, sizeInput, cudaMemcpyDeviceToHost));
    if (h_res == h_CPU) std::cout << "   Status: Correct\n\n";
    else std::cout << "   Status: INCORRECT!\n\n";
    
    cudaFree(d_A); cudaFree(d_R); cudaFree(d_A_Bitonic);
    return 0;
}
