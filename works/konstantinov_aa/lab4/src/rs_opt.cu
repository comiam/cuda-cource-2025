#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
#include <type_traits>

#define CUDA_CHECK(call) do {                          \
    cudaError_t err = call;                           \
    if (err != cudaSuccess) {                         \
        std::cerr << "CUDA error: "                   \
                  << cudaGetErrorString(err) << "\n"; \
        std::exit(1);                                 \
    }                                                 \
} while(0)

// ------------------ КОНСТАНТЫ / CONSTANTS ------------------
constexpr int BLOCK_SIZE = 256;
// 8 бит на проход (256 корзин)
constexpr int RADIX_BITS = 8;
constexpr int RADIX = 256;
constexpr int RADIX_MASK = 255;

// ------------------ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ / HELPERS ------------------

// Функция для извлечения "цифры" (байта) из числа
template <typename T>
__device__ __forceinline__ unsigned int get_digit(T x, int shift) {
    // Приводим к unsigned int, так как результат маски 0..255 всегда влезает в 32 бита
    return (unsigned int)((x >> shift) & RADIX_MASK);
}

// Warp-level scan helper (inclusive)
// Помощник для сканирования внутри варпа (32 потока)
// Использует __shfl_up_sync для обмена данными через регистры (без Shared Memory)
__device__ __forceinline__ unsigned int warp_inclusive_scan(unsigned int val) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
        unsigned int temp = __shfl_up_sync(mask, val, delta);
        if ((threadIdx.x & 31) >= delta) val += temp;
    }
    return val;
}



// 1. Histogram: Writes Transposed [Digit][Block]
// Паттерн: Map + Reduce (By Key)
//
// Задача: Посчитать, сколько раз каждая цифра встречается в каждом блоке.
// Вход: Исходный массив d_in.
// Выход: Матрица счетчиков d_counters размером [RADIX][NUM_BLOCKS].
//
// Оптимизация: Использует Shared Memory для атомарных операций, чтобы избежать коллизий в глобальной памяти.
// Запись происходит ТРАНСПОНИРОВАННО (по столбцам), чтобы на следующем шаге
// данные для одной цифры со всех блоков лежали подряд.
template <typename T>
__global__ void histogram_kernel(const T* d_in, unsigned int* d_counters, int n, int shift, int num_blocks) {
    __shared__ unsigned int local_hist[RADIX]; // Этот массив будет общим для всех потоков одного блока. Каждый блок имеет свою копию. (SM)

    int tid = threadIdx.x;
    if (tid < RADIX) local_hist[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        unsigned int digit = get_digit(d_in[idx], shift);
        // Reduce: Локальная редукция по ключу (цифре)
        atomicAdd(&local_hist[digit], 1); // увеличение счетчика для цифры
    }
    __syncthreads();

    if (tid < RADIX) {
        // Запись результата в транспонированном виде: [Digit][Block] в глоб. память
        d_counters[tid * num_blocks + blockIdx.x] = local_hist[tid];
    }
}

// 2. Parallel Prescan: 256 blocks, each scans one digit across all blocks
// Паттерн: Scan (Exclusive Prefix Sum)
//
// Задача: Превратить "количества" (счетчики) в "смещения" (адреса).
// Параллелизм: Каждый блок CUDA отвечает за ОДНУ цифру (GridSize = RADIX).
//
// Алгоритм:
// - Читает столбец матрицы d_counters (счетчики одной цифры со всех блоков).
// - Выполняет Exclusive Scan (Prefix Sum).
// - Результат d_offsets говорит каждому блоку, куда писать его элементы данной цифры.
// - Также вычисляет общее количество элементов данной цифры (d_digit_totals).
__global__ void prescan_kernel(unsigned int* d_counters, unsigned int* d_offsets, unsigned int* d_digit_totals, int num_blocks) {
    int digit = blockIdx.x; // 0..255
    int tid = threadIdx.x;
    
    unsigned int* my_counters = &d_counters[digit * num_blocks];
    unsigned int* my_offsets  = &d_offsets[digit * num_blocks];
    
    unsigned int carry = 0;
    
    // Блок обрабатывает длинный массив (num_blocks элементов) чанками по BLOCK_SIZE
    for (int offset = 0; offset < num_blocks; offset += BLOCK_SIZE) {
        int idx = offset + tid;
        unsigned int val = 0;
        if (idx < num_blocks) val = my_counters[idx];
        
        // 1. Warp Scan: Сканирование внутри варпа
        unsigned int warp_sum = warp_inclusive_scan(val);
        
        // 2. Block Scan: Объединение результатов варпов через Shared Memory
        __shared__ unsigned int warp_totals[BLOCK_SIZE / 32];
        int lane = tid % 32;
        int warp = tid / 32;
        if (lane == 31) warp_totals[warp] = warp_sum;
        __syncthreads();
        
        if (warp == 0) {
            unsigned int v = (tid < BLOCK_SIZE/32) ? warp_totals[tid] : 0;
            v = warp_inclusive_scan(v);
            if (tid < BLOCK_SIZE/32) warp_totals[tid] = v;
        }
        __syncthreads();
        
        unsigned int block_val = warp_sum;
        if (warp > 0) block_val += warp_totals[warp - 1];
        
        // 3. Exclusive Scan + Global Carry
        unsigned int res = block_val - val + carry;
        
        if (idx < num_blocks) my_offsets[idx] = res;
        
        // Обновление переноса (carry) для следующей итерации
        unsigned int chunk_sum = warp_totals[(BLOCK_SIZE/32) - 1];
        carry += chunk_sum;
        __syncthreads();
    }
    
    if (tid == 0) d_digit_totals[digit] = carry;
}

// 3. Finalize Prescan
// Паттерн: Scan (Exclusive)
//
// Задача: Вычислить глобальные базы для каждой цифры.
// Пример: Если цифр "0" всего 100 штук, то цифры "1" должны начинаться с индекса 100.
// Работает в 1 блоке, так как входных данных всего 256 элементов.
__global__ void prescan_finalize(unsigned int* d_digit_totals) {
    if (threadIdx.x >= RADIX) return;
    
    unsigned int val = d_digit_totals[threadIdx.x];
    
    // Простой Scan на одном варпе/блоке
    unsigned int warp_sum = warp_inclusive_scan(val);
    
    __shared__ unsigned int warp_totals[RADIX / 32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 31) warp_totals[warp] = warp_sum;
    __syncthreads();
    
    if (warp == 0) {
        unsigned int v = (threadIdx.x < RADIX/32) ? warp_totals[threadIdx.x] : 0;
        v = warp_inclusive_scan(v);
        if (threadIdx.x < RADIX/32) warp_totals[threadIdx.x] = v;
    }
    __syncthreads();
    
    unsigned int global_sum = warp_sum;
    if (warp > 0) global_sum += warp_totals[warp - 1];
    
    d_digit_totals[threadIdx.x] = global_sum - val; // Exclusive result
}

// 4. Scatter Kernel (Stable Version)
// Паттерн: Map + Local Scan + Scatter
//
// Задача: Переместить элементы на их финальные позиции.
// Ключевой момент: СТАБИЛЬНОСТЬ. Порядок одинаковых элементов должен сохраниться.
template <typename T>
__global__ void scatter_kernel(const T* d_in, T* d_out, 
                               const unsigned int* d_offsets, 
                               const unsigned int* d_digit_bases, 
                               int n, int shift, int num_blocks) {
    
    // Shared memory для хранения счетчиков цифр по варпам: [Warp][Digit]
    // Размер: 8 варпов * 256 цифр * 4 байта = 8 КБ (вмещается в стандартные 48 КБ)
    __shared__ unsigned int s_warp_counts[BLOCK_SIZE / 32][RADIX];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // Инициализация shared memory нулями
    // Каждый поток обнуляет ~8 элементов (2048 / 256)
    int total_smem = (BLOCK_SIZE / 32) * RADIX;
    for (int i = tid; i < total_smem; i += blockDim.x) {
        ((unsigned int*)s_warp_counts)[i] = 0;
    }
    __syncthreads();
    
    T val = 0;
    unsigned int digit = 0xFFFFFFFF; // Маркер для неактивных потоков (чтобы не мешали подсчету 0..255)
    bool active = (idx < n);

    // Map: Загрузка
    if (active) {
        val = d_in[idx];
        digit = get_digit(val, shift);
    }
   
    // 1. Warp-level counting (Считаем совпадения внутри варпа)
    // __match_any_sync возвращает маску потоков в варпе, у которых 'digit' совпадает
    unsigned int mask = __match_any_sync(0xFFFFFFFF, digit);
    
    // Ранг внутри варпа: сколько потоков с таким же digit имеют меньший lane ID
    unsigned int rank_in_warp = __popc(mask & ((1U << lane) - 1));
    
    // Общее количество таких элементов в текущем варпе
    unsigned int total_in_warp = __popc(mask);

    // 2. Агрегация по варпам через Shared Memory
    // Только первый встретившийся поток для каждой цифры в варпе пишет в SM
    // (rank_in_warp == 0 гарантирует, что пишет только один поток для каждой уникальной цифры в варпе)
    // Проверка digit < RADIX отсекает неактивные потоки
    if (digit < RADIX && rank_in_warp == 0) {
        s_warp_counts[warp_id][digit] = total_in_warp;
    }
    __syncthreads(); // Ждем, пока все варпы запишут свои счетчики

    if (!active) return;

    // 3. Подсчет глобального ранга в блоке
    // Складываем количества из всех предыдущих варпов + позиция внутри своего варпа
    unsigned int local_rank = rank_in_warp;
    #pragma unroll
    for (int w = 0; w < warp_id; ++w) {
        local_rank += s_warp_counts[w][digit];
    }

    // 4. Вычисление глобального адреса
    // GlobalPos = GlobalBase[digit] + BlockOffset[digit][block] + LocalRank
    unsigned int block_offset = d_offsets[digit * num_blocks + blockIdx.x];
    unsigned int global_base = d_digit_bases[digit];
    unsigned int global_pos = global_base + block_offset + local_rank;

    // 5. Scatter: Запись
    d_out[global_pos] = val;
}

// ------------------ HOST ------------------
template <typename T>
void radix_sort(T* d_in, T* d_tmp, int n) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    unsigned int *d_counters, *d_offsets, *d_digit_totals; //гистограмма, смещение, глобальные суммы(общее количество каждой цифры размеров RADIX)
    
    size_t counters_size = num_blocks * RADIX * sizeof(unsigned int);
    
    CUDA_CHECK(cudaMalloc(&d_counters, counters_size));
    CUDA_CHECK(cudaMalloc(&d_offsets, counters_size));
    CUDA_CHECK(cudaMalloc(&d_digit_totals, RADIX * sizeof(unsigned int)));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(num_blocks);

    // Главный цикл: проход по байтам (8 бит).
    // Для int (32 бит) = 4 прохода.
    // Для long long (64 бит) = 8 проходов.
    int num_bits = sizeof(T) * 8;
    for (int shift = 0; shift < num_bits; shift += RADIX_BITS) {
        // Шаг 1: Статистика
        CUDA_CHECK(cudaMemset(d_counters, 0, counters_size));
        histogram_kernel<<<grid, block>>>(d_in, d_counters, n, shift, num_blocks);
        
        // Шаг 2: Скан счетчиков
        prescan_kernel<<<RADIX, BLOCK_SIZE>>>(d_counters, d_offsets, d_digit_totals, num_blocks);
        
        // Шаг 3: Глобальные базы
        prescan_finalize<<<1, RADIX>>>(d_digit_totals); 
        
        // Шаг 4: Рассеивание
        scatter_kernel<<<grid, block>>>(d_in, d_tmp, d_offsets, d_digit_totals, n, shift, num_blocks);
        
        // Меняем буферы местами
        std::swap(d_in, d_tmp);
    }

    CUDA_CHECK(cudaFree(d_counters));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_digit_totals));
}

// ------------------ TEST RUNNER ------------------
template <typename T>
void run_test(const char* label, int N) {
    std::cout << "=== Testing " << label << " with N=" << N << " ===\n";

    // 1. Host Data Gen
    std::vector<T> h(N);
    std::mt19937_64 gen(42);
    for (int i = 0; i < N; ++i) {
        if (sizeof(T) == 8) h[i] = (T)gen(); 
        else h[i] = (T)(unsigned int)gen();
    }
    
    std::vector<T> h_ref = h;

    // 2. Prepare for GPU (Handle Signedness)
    // Radix Sort работает корректно только с unsigned ключами.
    // Чтобы сортировать знаковые (signed), нужно инвертировать знаковый бит.
    using UnsignedT = typename std::make_unsigned<T>::type;
    UnsignedT sign_mask = (UnsignedT)1 << (sizeof(T) * 8 - 1);
    
    std::vector<UnsignedT> hu(N);
    for(int i=0; i<N; ++i) {
        hu[i] = ((UnsignedT)h[i]) ^ sign_mask;
    }

    UnsignedT *d_in, *d_tmp; // Рабочий массив для сортировки и временный
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(UnsignedT)));
    CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(UnsignedT)));
    CUDA_CHECK(cudaMemcpy(d_in, hu.data(), N * sizeof(UnsignedT), cudaMemcpyHostToDevice));

    // 3. Run Custom Sort
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    radix_sort(d_in, d_tmp, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_custom;
    cudaEventElapsedTime(&ms_custom, start, stop);

    // 4. Verify Custom Sort
    CUDA_CHECK(cudaMemcpy(hu.data(), d_in, N * sizeof(UnsignedT), cudaMemcpyDeviceToHost));
    // Restore sign bit
    for(int i=0; i<N; ++i) {
        h[i] = (T)(hu[i] ^ sign_mask);
    }

    // Run CPU Sort
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(h_ref.begin(), h_ref.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_cpu = end_cpu - start_cpu;

    bool ok = (h == h_ref);

    // 5. Run Thrust (for benchmark)
    thrust::device_vector<T> d_thrust = h_ref; 
    // Regenerate to be fair
    std::mt19937_64 gen2(42);
    std::vector<T> h_thrust_input(N);
    for (int i = 0; i < N; ++i) {
        if (sizeof(T) == 8) h_thrust_input[i] = (T)gen2(); 
        else h_thrust_input[i] = (T)(unsigned int)gen2();
    }
    d_thrust = h_thrust_input;

    cudaEventRecord(start);
    thrust::sort(d_thrust.begin(), d_thrust.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_thrust;
    cudaEventElapsedTime(&ms_thrust, start, stop);

    // Output
    std::cout << "\nResults for " << label << ":\n";
    std::cout << "Correctness:        " << (ok ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "Time CPU std::sort: " << ms_cpu.count() / 1000.0 << " s\n";
    std::cout << "Time GPU Custom:    " << ms_custom / 1000.0 << " s\n";
    std::cout << "Time GPU Thrust:    " << ms_thrust / 1000.0 << " s\n";
    std::cout << "Speedup vs CPU:     " << ms_cpu.count() / ms_custom << "x\n";
    std::cout << "Speedup vs Thrust:  " << ms_thrust / ms_custom << "x\n";
    std::cout << "------------------------------------------------\n\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_tmp));
}

int main() {
    int N = 100000000;
    
    // Test 32-bit (int)
    run_test<int>("32-bit Integer (int)", N);
    
    // Test 64-bit (long long)
    run_test<long long>("64-bit Integer (long long)", N);

    return 0;
}
