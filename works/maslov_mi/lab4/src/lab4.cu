#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#define HIST_BINS 256

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::cerr << cudaGetErrorString(e) << std::endl; exit(1);} } while(0)

static void print_sample(const std::vector<uint32_t>& v, const char* label, size_t max_elems = 20)
{
    size_t cnt = std::min(max_elems, v.size());
    std::cout << label << " (first " << cnt << "): ";
    for (size_t i = 0; i < cnt; i++)
        std::cout << v[i] << (i + 1 == cnt ? "" : " ");
    std::cout << "\n";
}

// static void print_hist_sample(const std::vector<uint32_t>& h, const char* label, size_t max_bins = 16)
// {
//     size_t cnt = std::min(max_bins, h.size());
//     std::cout << label << " (bins 0.." << cnt - 1 << "): ";
//     for (size_t i = 0; i < cnt; i++)
//         std::cout << h[i] << (i + 1 == cnt ? "" : " ");
//     std::cout << "\n";
// }

__global__
void histogram_atomic(const uint32_t* data, int n, unsigned int* hist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        unsigned int bin = data[i] & 0xFFu;
        atomicAdd(&hist[bin], 1);
    }
}

static float histogramGPU(const std::vector<uint32_t>& data, std::vector<uint32_t>& histOut)
{
    int n = static_cast<int>(data.size());
    histOut.assign(HIST_BINS, 0);
    if (n == 0) return 0.0f;

    uint32_t* d_data = nullptr;
    unsigned int* d_hist = nullptr;

    CHECK(cudaMalloc(&d_data, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_hist, HIST_BINS * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(unsigned int)));

    int threads = 256;
    int blocks = std::min(65535, (n + threads - 1) / threads);
    // size_t shmem = HIST_BINS * sizeof(unsigned int);

    cudaEvent_t evStart, evStop;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evStop));

    CHECK(cudaEventRecord(evStart));
    histogram_atomic<<<blocks, threads>>>(d_data, n, d_hist);
    CHECK(cudaEventRecord(evStop));
    CHECK(cudaEventSynchronize(evStop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    CHECK(cudaMemcpy(histOut.data(), d_hist, HIST_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_hist);

    return ms;
}

#define RADIX_BITS 8
#define RADIX_BUCKETS (1 << RADIX_BITS)  // 256 корзин
#define RADIX_MASK (RADIX_BUCKETS - 1)
#define SORT_THREADS 256
#define WARPS_PER_BLOCK (SORT_THREADS / 32)

// Вычисление гистограмм по разрядам для каждого блока
// Каждый варп считает свою локальную гистограмму, чтобы избежать конфликтов при записи
__global__
void radix_histogram(const uint32_t* in, int n, unsigned int* histograms, 
                     int shift, int elemsPerBlock)
{
    // Выделяем память под гистограмму для каждого варпа
    __shared__ unsigned int shist[WARPS_PER_BLOCK][RADIX_BUCKETS];
    
    int t = threadIdx.x;
    int laneId = t & 31;
    int warpId = t >> 5;
    int blockId = blockIdx.x;
    
    // Инициализация shared memory
    unsigned int* shist_flat = &shist[0][0];
    for (int i = t; i < WARPS_PER_BLOCK * RADIX_BUCKETS; i += blockDim.x)
        shist_flat[i] = 0;
    __syncthreads();
    
    // Обработка тайла элементов данного блока
    int start = blockId * elemsPerBlock;
    int end = min(start + elemsPerBlock, n);
    
    unsigned int* my_warp_hist = shist[warpId];

    for (int i = start + t; i < end; i += blockDim.x) {
        unsigned int digit = (in[i] >> shift) & RADIX_MASK;
        
        // нулевой поток варпа собирает значения остальных и обновляет счетчики
        unsigned int mask = __activemask();
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            unsigned int d = __shfl_sync(mask, digit, k);
            // Если k-й поток активен, нулевой поток учитывает его значение
            if (laneId == 0 && (mask & (1u << k))) {
                my_warp_hist[d]++;
            }
        }
    }
    __syncthreads();
    
    // Суммируем результаты всех варпов и пишем итог в глобальную память
    int numBlocks = gridDim.x;
    for (int b = t; b < RADIX_BUCKETS; b += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            sum += shist[w][b];
        }
        histograms[b * numBlocks + blockId] = sum;
    }
}

// Префиксная сумма по всем гистограммам блоков (на GPU, без возврата на CPU)
// Каждый поток обрабатывает одну корзину разряда через все блоки
__global__
void histogram_prefix_sum(unsigned int* histograms, unsigned int* digit_totals, int numBlocks)
{
    int digit = blockIdx.x * blockDim.x + threadIdx.x;
    if (digit >= RADIX_BUCKETS) return;
    
    unsigned int* row = histograms + digit * numBlocks;
    unsigned int sum = 0;
    
    // Последовательный префикс для этого разряда по всем блокам
    for (int b = 0; b < numBlocks; b++) {
        unsigned int val = row[b];
        row[b] = sum;  // Сохраняем префикс
        sum += val;
    }
    
    // Сохраняем общий счётчик для этого разряда
    digit_totals[digit] = sum;
}

// Вычисление глобальных стартовых смещений для каждого разряда
__global__
void compute_digit_offsets(unsigned int* digit_totals, unsigned int* digit_offsets)
{
    __shared__ unsigned int temp[RADIX_BUCKETS];
    
    int t = threadIdx.x;
    
    // Загрузка общих сумм
    if (t < RADIX_BUCKETS)
        temp[t] = digit_totals[t];
    else
        temp[t] = 0;
    __syncthreads();
    
    // Cкан
    for (int d = 1; d < RADIX_BUCKETS; d <<= 1) {
        int idx = (t + 1) * d * 2 - 1;
        if (idx < RADIX_BUCKETS)
            temp[idx] += temp[idx - d];
        __syncthreads();
    }
    
    // Последний элемент в 0 для эксклюзивного варианта
    if (t == 0) 
        temp[RADIX_BUCKETS - 1] = 0;
    __syncthreads();
    
    // Спуск
    for (int d = RADIX_BUCKETS / 2; d >= 1; d >>= 1) {
        int idx = (t + 1) * d * 2 - 1;
        if (idx < RADIX_BUCKETS) {
            unsigned int tmp = temp[idx - d];
            temp[idx - d] = temp[idx];
            temp[idx] += tmp;
        }
        __syncthreads();
    }
    
    // Сохраняем смещения
    if (t < RADIX_BUCKETS)
        digit_offsets[t] = temp[t];
}

// Спуск: раскидывание ключей в выход с учётом посчитанных смещений
// Cначала ранжируем ключи локально в shared, затем записываем
__global__
void radix_scatter(const uint32_t* in, uint32_t* out, int n,
                   const unsigned int* histograms, 
                   const unsigned int* digit_offsets,
                   int shift, int numBlocks, int elemsPerBlock)
{
    __shared__ unsigned int soffset[RADIX_BUCKETS];    // смещение для каждого разряда
    __shared__ unsigned int scount[RADIX_BUCKETS];    // счётчик для текущего тайла
    __shared__ uint16_t sranks[SORT_THREADS];         // ранг для текущего тайла
    __shared__ uint8_t sdigits[SORT_THREADS];         // разряд для текущего тайла
    
    int t = threadIdx.x;
    int blockId = blockIdx.x;
    
    // Загрузка стартовых смещений для каждого разряда в этом блоке
    for (int b = t; b < RADIX_BUCKETS; b += blockDim.x) {
        unsigned int blockPrefix = histograms[b * numBlocks + blockId];
        unsigned int digitStart = digit_offsets[b];
        soffset[b] = digitStart + blockPrefix;
    }
    __syncthreads();
    
    int start = blockId * elemsPerBlock;
    int end = min(start + elemsPerBlock, n);
    
    // Обработка тайлами по blockDim.x
    for (int tile = start; tile < end; tile += blockDim.x) {
        // Сброс счётчиков для текущего тайла
        for (int b = t; b < RADIX_BUCKETS; b += blockDim.x)
            scount[b] = 0;
        __syncthreads();
        
        int i = tile + t;
        bool valid = (i < end);
        uint32_t key = 0;
        unsigned int digit = 0;
        
        if (valid) {
            key = in[i];
            digit = (key >> shift) & RADIX_MASK;
        }
        sdigits[t] = (uint8_t)digit;
        __syncthreads();
        
        // Подсчёт ранга элемента: сколько предыдущих с тем же разрядом
        if (valid) {
            unsigned int rank = 0;
            for (int j = 0; j < t; j++) {
                if (sdigits[j] == digit)
                    rank++;
            }
            sranks[t] = (uint16_t)rank;
            
            // Подсчёт общего количества для этого разряда
            atomicAdd(&scount[digit], 1);
        }
        __syncthreads();
        
        // Запись ключей в output
        if (valid) {
            unsigned int pos = soffset[digit] + sranks[t];
            out[pos] = key;
        }
        __syncthreads();
        
        // Обновление смещений для следующего тайла
        for (int b = t; b < RADIX_BUCKETS; b += blockDim.x)
            soffset[b] += scount[b];
        __syncthreads();
    }
}

void radixSort32(std::vector<uint32_t>& arr)
{
    int n = arr.size();
    if (n == 0) return;

    uint32_t *d_in, *d_out;
    unsigned int *d_histograms, *d_digit_totals, *d_digit_offsets;

    CHECK(cudaMalloc(&d_in,  n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_in, arr.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Определяем число блоков и количество элементов на блок
    int threads = SORT_THREADS;
    int elemsPerBlock = threads * 4;  // Один блок обрабатывает 4x threads элементов
    int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;
    numBlocks = std::max(1, std::min(numBlocks, 1024));  // Ограничиваем число блоков
    elemsPerBlock = (n + numBlocks - 1) / numBlocks;     // Пересчитываем
    
    // Выделение памяти под гистограммы
    // histograms: RADIX_BUCKETS x numBlocks (column-major)
    CHECK(cudaMalloc(&d_histograms, RADIX_BUCKETS * numBlocks * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_digit_totals, RADIX_BUCKETS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_digit_offsets, RADIX_BUCKETS * sizeof(unsigned int)));

    // 4 прохода для 32-битных ключей (8 бит на проход)
    for (int pass = 0; pass < 4; pass++)
    {
        int shift = pass * RADIX_BITS;
        
        // 1. Подъём: вычислить гистограммы по блокам
        radix_histogram<<<numBlocks, threads>>>(d_in, n, d_histograms, shift, elemsPerBlock);
        
        // 2. Префиксная сумма по гистограммам (полностью на GPU)
        // Для каждого разряда — префикс по всем блокам
        int prefixThreads = RADIX_BUCKETS;  // 256 threads, one per digit
        histogram_prefix_sum<<<1, prefixThreads>>>(d_histograms, d_digit_totals, numBlocks);
        
        // 3. Вычислить глобальные смещения разрядов (эксклюзивный префикс общих сумм)
        compute_digit_offsets<<<1, RADIX_BUCKETS>>>(d_digit_totals, d_digit_offsets);
        
        // 4. Спуск: раскидать ключи в выход
        radix_scatter<<<numBlocks, threads>>>(d_in, d_out, n, 
                                               d_histograms, d_digit_offsets,
                                               shift, numBlocks, elemsPerBlock);
        
        // Меняем буферы для следующего прохода
        std::swap(d_in, d_out);
    }

    // Копируем результат обратно (d_in - результат после swap)
    CHECK(cudaMemcpy(arr.data(), d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_histograms);
    cudaFree(d_digit_totals);
    cudaFree(d_digit_offsets);
}


int main(int argc, char** argv)
{
    int N = 1000000;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    std::vector<uint32_t> data(N);
    for (int i = 0; i < N; i++) data[i] = rand();

    // print_sample(data, "Input sample");

    auto histCpu = std::vector<uint32_t>(HIST_BINS, 0);
    auto h0 = std::chrono::high_resolution_clock::now();
    for (uint32_t v : data)
        histCpu[v & 0xFFu]++;
    auto h1 = std::chrono::high_resolution_clock::now();
    double cpu_hist_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();

    std::vector<uint32_t> histGpu;
    float gpu_hist_ms = histogramGPU(data, histGpu);

    bool hist_ok = (histCpu == histGpu);
    if (!hist_ok)
        std::cout << "Histogram mismatch between CPU and GPU!\n";

    auto hostVec = data;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(hostVec.begin(), hostVec.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto gpuRadixVec = data;
    cudaEvent_t evStartRadix, evStopRadix;
    CHECK(cudaEventCreate(&evStartRadix));
    CHECK(cudaEventCreate(&evStopRadix));

    CHECK(cudaEventRecord(evStartRadix));
    radixSort32(gpuRadixVec);
    CHECK(cudaEventRecord(evStopRadix));
    CHECK(cudaEventSynchronize(evStopRadix));

    float gpu_ms_radix = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms_radix, evStartRadix, evStopRadix));

    cudaEventDestroy(evStartRadix);
    cudaEventDestroy(evStopRadix);

    // print_sample(gpuRadixVec, "After radix sort");

    auto gpuThrustHost = data;
    thrust::device_vector<uint32_t> d_thrust(gpuThrustHost.begin(), gpuThrustHost.end());
    cudaEvent_t evStartThrust, evStopThrust;
    CHECK(cudaEventCreate(&evStartThrust));
    CHECK(cudaEventCreate(&evStopThrust));

    CHECK(cudaEventRecord(evStartThrust));
    thrust::sort(d_thrust.begin(), d_thrust.end());
    CHECK(cudaEventRecord(evStopThrust));
    CHECK(cudaEventSynchronize(evStopThrust));

    float gpu_ms_thrust = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms_thrust, evStartThrust, evStopThrust));

    cudaEventDestroy(evStartThrust);
    cudaEventDestroy(evStopThrust);

    thrust::copy(d_thrust.begin(), d_thrust.end(), gpuThrustHost.begin());
    // print_sample(gpuThrustHost, "After thrust::sort");

    bool okRadix = std::is_sorted(gpuRadixVec.begin(), gpuRadixVec.end());
    bool okThrust = std::is_sorted(gpuThrustHost.begin(), gpuThrustHost.end());
    if (!okRadix || !okThrust)
        std::cout << "Sort FAILED! (radix=" << okRadix << ", thrust=" << okThrust << ")\n";

    double speedupRadix = cpu_ms / gpu_ms_radix;
    double speedupThrust = cpu_ms / gpu_ms_thrust;
    double diffThrustRadix = gpu_ms_radix / gpu_ms_thrust;

    std::cout << "\nArray size: " << N << "\n\n";
    std::cout << "CPU sort time: " << std::fixed << std::setprecision(4) << cpu_ms / 1000.0 << " s\n";
    std::cout << "GPU Radix Sort time: " << std::fixed << std::setprecision(4) << gpu_ms_radix / 1000.0 << " s\n";
    std::cout << "GPU thrust::sort time: " << std::fixed << std::setprecision(4) << gpu_ms_thrust / 1000.0 << " s\n";
    std::cout << "Speedup (Radix vs CPU): " << std::fixed << std::setprecision(2) << speedupRadix << "x\n";
    std::cout << "Speedup (thrust::sort vs CPU): " << std::fixed << std::setprecision(2) << speedupThrust << "x\n";
    std::cout << "Speedup (thrust::sort vs Radix): " << std::fixed << std::setprecision(2) << diffThrustRadix << "x\n";
}
