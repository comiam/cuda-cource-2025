#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS) // 16 бакетов
#define RADIX_MASK (RADIX_SIZE - 1)  // 0xF

#define cudaCheckError(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__device__ unsigned int get_digit(unsigned int x, int shift)
{
    return (x >> shift) & RADIX_MASK;
}

__device__ unsigned int get_digit_64(unsigned long long x, int shift)
{
    return (unsigned int)((x >> shift) & RADIX_MASK);
}

// ядро гистограммы - подсчитывает количество вхождений каждой цифры в блоках
__global__ void histogram_kernel(const unsigned int *input, unsigned int *hist, int n, int shift)
{
    __shared__ unsigned int local_hist[RADIX_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // инициализировать разделяемую память
    if (tid < RADIX_SIZE)
    {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // подсчитать цифры в диапазоне этого потока
    if (idx < n)
    {
        unsigned int digit = get_digit(input[idx], shift);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // записать в глобальную память
    if (tid < RADIX_SIZE)
    {
        hist[bid * RADIX_SIZE + tid] = local_hist[tid];
    }
}

// warp-level инклюзивное сканирование
__device__ unsigned int warp_inclusive_scan(unsigned int val)
{
    unsigned int mask = 0xffffffff;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
    {
        unsigned int temp = __shfl_up_sync(mask, val, offset);
        val += temp;
    }
    return val;
}

// warp-level эксклюзивное сканирование
__device__ unsigned int warp_exclusive_scan(unsigned int val)
{
    unsigned int incl = warp_inclusive_scan(val);
    unsigned int excl = __shfl_up_sync(0xffffffff, incl, 1);
    return excl;
}

// histogram kernel for 64-bit values
__global__ void histogram_kernel_64(const unsigned long long *input, unsigned int *hist, int n, int shift)
{
    __shared__ unsigned int local_hist[RADIX_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    if (tid < RADIX_SIZE)
    {
        local_hist[tid] = 0;
    }
    __syncthreads();

    if (idx < n)
    {
        unsigned int digit = get_digit_64(input[idx], shift);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // записать в глобальную память
    if (tid < RADIX_SIZE)
    {
        hist[bid * RADIX_SIZE + tid] = local_hist[tid];
    }
}

// block-level prefix sum
__global__ void prefix_sum_kernel(unsigned int *hist, unsigned int *prefix_sum, int num_blocks)
{
    if (threadIdx.x == 0)
    {
        // обработать каждую цифру последовательно
        for (int digit = 0; digit < RADIX_SIZE; digit++)
        {
            unsigned int sum = 0;
            for (int b = 0; b < num_blocks; b++)
            {
                unsigned int count = hist[b * RADIX_SIZE + digit];
                prefix_sum[b * RADIX_SIZE + digit] = sum; // это начальная позиция для этого блока и цифры
                sum += count;
            }
        }
    }
}

// оптимизированное блочное префиксное суммирование с использованием warp операций
__global__ void prefix_sum_kernel_optimized(unsigned int *hist, unsigned int *prefix_sum, int num_blocks)
{
    __shared__ unsigned int temp[RADIX_SIZE];

    int tid = threadIdx.x;

    // обработать каждую цифру параллельно
    if (tid < RADIX_SIZE)
    {
        unsigned int digit = tid;
        unsigned int sum = 0;

        // вычислить префиксную сумму для этой цифры по всем блокам
        for (int b = 0; b < num_blocks; b++)
        {
            unsigned int count = hist[b * RADIX_SIZE + digit];
            prefix_sum[b * RADIX_SIZE + digit] = sum;
            sum += count;
        }
    }
}

__global__ void scatter_kernel(const unsigned int *input, unsigned int *output,
                               const unsigned int *prefix_sum, int n, int shift, int num_blocks)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // использовать shared memory для временного хранения значений и их цифр
    __shared__ unsigned int temp_values[BLOCK_SIZE];
    __shared__ unsigned int temp_digits[BLOCK_SIZE];

    // загрузить данные в shared memory
    if (idx < n)
    {
        temp_values[tid] = input[idx];
        temp_digits[tid] = get_digit(input[idx], shift);
    }
    else
    {
        temp_values[tid] = 0;
        temp_digits[tid] = 0;
    }
    __syncthreads();

    // использовать warp-level синхронизацию для подсчета количества элементов с каждой цифрой
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // использовать shared memory для отслеживания количества элементов с одинаковой цифрой в каждом warp
    __shared__ unsigned int warp_digit_counts[BLOCK_SIZE / 32][RADIX_SIZE];

    // инициализировать shared memory для каждого warp
    if (lane_id == 0)
    {
        for (int i = 0; i < RADIX_SIZE; i++)
        {
            warp_digit_counts[warp_id][i] = 0;
        }
    }
    __syncthreads();

    // использовать атомарную операцию только для подсчета в пределах warp
    // но только один раз на warp для каждой цифры
    if (idx < n)
    {
        unsigned int digit = temp_digits[tid];
        atomicAdd(&warp_digit_counts[warp_id][digit], 1);
    }
    __syncthreads();

    // теперь вычисляем глобальные позиции для каждого элемента
    if (idx < n)
    {
        unsigned int value = temp_values[tid];
        unsigned int digit = temp_digits[tid];

        // вычислить позицию с использованием префиксной суммы
        unsigned int block_offset = bid * RADIX_SIZE + digit;
        unsigned int global_offset = prefix_sum[block_offset];

        // вычислить локальную позицию внутри блока
        unsigned int local_pos = 0;
        for (int w = 0; w < warp_id; w++)
        {
            local_pos += warp_digit_counts[w][digit];
        }

        // добавить локальное смещение в пределах warp
        for (int t = 0; t < lane_id; t++)
        {
            if (t < BLOCK_SIZE && temp_digits[t + warp_id * 32] == digit)
            {
                local_pos++;
            }
        }

        // итоговая позиция
        unsigned int final_pos = global_offset + local_pos;
        output[final_pos] = value;
    }
}

// 64-bit
__global__ void scatter_kernel_64(const unsigned long long *input, unsigned long long *output,
                                  const unsigned int *prefix_sum, int n, int shift, int num_blocks)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // использовать shared memory для временного хранения значений и их цифр
    __shared__ unsigned long long temp_values[BLOCK_SIZE];
    __shared__ unsigned int temp_digits[BLOCK_SIZE];

    // загрузить данные в shared memory
    if (idx < n)
    {
        temp_values[tid] = input[idx];
        temp_digits[tid] = get_digit_64(input[idx], shift);
    }
    else
    {
        temp_values[tid] = 0;
        temp_digits[tid] = 0;
    }
    __syncthreads();

    // использовать warp-level синхронизацию для подсчета количества элементов с каждой цифрой
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // использовать shared memory для отслеживания количества элементов с одинаковой цифрой в каждом warp
    __shared__ unsigned int warp_digit_counts[BLOCK_SIZE / 32][RADIX_SIZE];

    // инициализировать shared memory для каждого warp
    if (lane_id == 0)
    {
        for (int i = 0; i < RADIX_SIZE; i++)
        {
            warp_digit_counts[warp_id][i] = 0;
        }
    }
    __syncthreads();

    // использовать атомарную операцию только для подсчета в пределах warp
    // но только один раз на warp для каждой цифры
    if (idx < n)
    {
        unsigned int digit = temp_digits[tid];
        atomicAdd(&warp_digit_counts[warp_id][digit], 1);
    }
    __syncthreads();

    // теперь вычисляем глобальные позиции для каждого элемента
    if (idx < n)
    {
        unsigned long long value = temp_values[tid];
        unsigned int digit = temp_digits[tid];

        // вычислить позицию с использованием префиксной суммы
        unsigned int block_offset = bid * RADIX_SIZE + digit;
        unsigned int global_offset = prefix_sum[block_offset];

        // вычислить локальную позицию внутри блока
        unsigned int local_pos = 0;
        for (int w = 0; w < warp_id; w++)
        {
            local_pos += warp_digit_counts[w][digit];
        }

        // добавить локальное смещение в пределах warp
        for (int t = 0; t < lane_id; t++)
        {
            if (t < BLOCK_SIZE && temp_digits[t + warp_id * 32] == digit)
            {
                local_pos++;
            }
        }

        // итоговая позиция
        unsigned int final_pos = global_offset + local_pos;
        output[final_pos] = value;
    }
}

// ядро для изменения знакового бита 32-битных целых
__global__ void flip_sign_bit_kernel(unsigned int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] ^= 0x80000000; // меняем знак бит
    }
}

void manualRadixSortInt(int *h_data, int n)
{
    unsigned int *d_input, *d_output, *d_temp;
    unsigned int *d_hist, *d_prefix_sum;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int min_blocks_for_efficiency = 256; // эффективное количество блоков для современных GPU
    num_blocks = std::max(num_blocks, min_blocks_for_efficiency);

    // создать CUDA stream для вычислений
    cudaStream_t stream_compute;
    cudaStreamCreate(&stream_compute);

    // выделить память на устройстве
    cudaCheckError(cudaMalloc(&d_input, n * sizeof(unsigned int)));
    cudaCheckError(cudaMalloc(&d_output, n * sizeof(unsigned int)));
    cudaCheckError(cudaMalloc(&d_hist, num_blocks * RADIX_SIZE * sizeof(unsigned int)));
    cudaCheckError(cudaMalloc(&d_prefix_sum, num_blocks * RADIX_SIZE * sizeof(unsigned int)));

    // копировать входные данные на устройство
    cudaCheckError(cudaMemcpy(d_input, h_data, n * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // инвертировать знаковый бит для правильной обработки знаковых целых
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);
    flip_sign_bit_kernel<<<grid, block, 0, stream_compute>>>(d_input, n);
    cudaCheckError(cudaGetLastError());

    // главный цикл поразрядной сортировки - обрабатывать по 4 бита за раз
    for (int shift = 0; shift < 32; shift += RADIX_BITS)
    {
        // шаг гистограммы
        histogram_kernel<<<num_blocks, BLOCK_SIZE, 0, stream_compute>>>(d_input, d_hist, n, shift);
        cudaCheckError(cudaGetLastError());

        // шаг префиксной суммы
        prefix_sum_kernel_optimized<<<1, RADIX_SIZE, 0, stream_compute>>>(d_hist, d_prefix_sum, num_blocks);
        cudaCheckError(cudaGetLastError());

        // шаг рассеивания
        scatter_kernel<<<num_blocks, BLOCK_SIZE, 0, stream_compute>>>(d_input, d_output, d_prefix_sum, n, shift, num_blocks);
        cudaCheckError(cudaGetLastError());

        // поменять местами вход и выход
        d_temp = d_input;
        d_input = d_output;
        d_output = d_temp;
    }

    // вернуть знаковый бит, чтобы восстановить правильные значения
    flip_sign_bit_kernel<<<grid, block, 0, stream_compute>>>(d_input, n);
    cudaCheckError(cudaGetLastError());

    // синхронизировать stream перед копированием результата
    cudaStreamSynchronize(stream_compute);

    // скопировать результат обратно на хост
    cudaCheckError(cudaMemcpy(h_data, d_input, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // уничтожить stream
    cudaStreamDestroy(stream_compute);

    // освободить память устройства
    cudaCheckError(cudaFree(d_input));
    cudaCheckError(cudaFree(d_output));
    cudaCheckError(cudaFree(d_hist));
    cudaCheckError(cudaFree(d_prefix_sum));
}

// ядро для изменения знакового бита 64-битных целых
__global__ void flip_sign_bit_kernel_64(unsigned long long *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] ^= 0x8000000000000000ULL; // меняем знак бит
    }
}

void manualRadixSortLong(long long *h_data, int n)
{
    unsigned long long *d_input, *d_output, *d_temp;
    unsigned int *d_hist, *d_prefix_sum;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int min_blocks_for_efficiency = 256; // эффективное количество блоков для современных GPU
    num_blocks = std::max(num_blocks, min_blocks_for_efficiency);

    // создать CUDA stream для вычислений
    cudaStream_t stream_compute;
    cudaStreamCreate(&stream_compute);

    // выделить память на устройстве
    cudaCheckError(cudaMalloc(&d_input, n * sizeof(unsigned long long)));
    cudaCheckError(cudaMalloc(&d_output, n * sizeof(unsigned long long)));
    cudaCheckError(cudaMalloc(&d_hist, num_blocks * RADIX_SIZE * sizeof(unsigned int)));
    cudaCheckError(cudaMalloc(&d_prefix_sum, num_blocks * RADIX_SIZE * sizeof(unsigned int)));

    // копировать входные данные на устройство
    cudaCheckError(cudaMemcpy(d_input, h_data, n * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // инвертировать знаковый бит для правильной обработки знаковых целых
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);
    flip_sign_bit_kernel_64<<<grid, block, 0, stream_compute>>>(d_input, n);
    cudaCheckError(cudaGetLastError());

    // главный цикл поразрядной сортировки - обрабатывать по 4 бита за раз
    for (int shift = 0; shift < 64; shift += RADIX_BITS)
    {
        // шаг гистограммы
        histogram_kernel_64<<<num_blocks, BLOCK_SIZE, 0, stream_compute>>>(d_input, d_hist, n, shift);
        cudaCheckError(cudaGetLastError());

        // шаг префиксной суммы
        prefix_sum_kernel_optimized<<<1, RADIX_SIZE, 0, stream_compute>>>(d_hist, d_prefix_sum, num_blocks);
        cudaCheckError(cudaGetLastError());

        // шаг рассеивания
        scatter_kernel_64<<<num_blocks, BLOCK_SIZE, 0, stream_compute>>>(d_input, d_output, d_prefix_sum, n, shift, num_blocks);
        cudaCheckError(cudaGetLastError());

        // поменять местами вход и выход
        d_temp = d_input;
        d_input = d_output;
        d_output = d_temp;
    }

    // вернуть знаковый бит, чтобы восстановить правильные значения
    flip_sign_bit_kernel_64<<<grid, block, 0, stream_compute>>>(d_input, n);
    cudaCheckError(cudaGetLastError());

    // синхронизировать stream перед копированием результата
    cudaStreamSynchronize(stream_compute);

    // скопировать результат обратно на хост
    cudaCheckError(cudaMemcpy(h_data, d_input, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // уничтожить stream
    cudaStreamDestroy(stream_compute);

    // освободить память устройства
    cudaCheckError(cudaFree(d_input));
    cudaCheckError(cudaFree(d_output));
    cudaCheckError(cudaFree(d_hist));
    cudaCheckError(cudaFree(d_prefix_sum));
}

// thrust
void radixSortInt(int *h_data, int n)
{
    thrust::device_vector<int> d_data(h_data, h_data + n);
    thrust::sort(d_data.begin(), d_data.end());
    thrust::copy(d_data.begin(), d_data.end(), h_data);
}

void radixSortLong(long long *h_data, int n)
{
    thrust::device_vector<long long> d_data(h_data, h_data + n);
    thrust::sort(d_data.begin(), d_data.end());
    thrust::copy(d_data.begin(), d_data.end(), h_data);
}

// функция проверки упорядоченности
bool verify_sorted(int *arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        if (arr[i] < arr[i - 1])
        {
            return false;
        }
    }
    return true;
}

// функция проверки упорядоченности для 64-bit
bool verify_sorted_long(long long *arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        if (arr[i] < arr[i - 1])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    printf("Testing 32-bit signed integers\n");

    // тестирование на разных размерах массивов
    int sizes[] = {100000, 1000000, 10000000};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < numSizes; s++)
    {
        int n = sizes[s];
        printf("\nTesting with array size: %d\n", n);

        // выделение памяти на хосте
        int *h_data_radix = new int[n];
        int *h_data_thrust = new int[n];
        int *h_data_manual = new int[n];
        int *h_data_std = new int[n];

        // генерация случайных данных, включая отрицательные числа
        srand(42);
        for (int i = 0; i < n; i++)
        {
            h_data_radix[i] = rand() - RAND_MAX / 2;
            h_data_thrust[i] = h_data_radix[i];
            h_data_manual[i] = h_data_radix[i];
            h_data_std[i] = h_data_radix[i];
        }

        // замер времени для поразрядной сортировки (реализованной через thrust)
        auto start = std::chrono::high_resolution_clock::now();
        radixSortInt(h_data_radix, n);
        auto end = std::chrono::high_resolution_clock::now();
        double radix_time = std::chrono::duration<double, std::milli>(end - start).count();

        // проверка результатов
        bool radix_correct = verify_sorted(h_data_radix, n);
        printf("Thrust Radix Sort: %.3f ms\n", radix_time);

        // замер времени для ручной реализации поразрядной сортировки
        start = std::chrono::high_resolution_clock::now();
        manualRadixSortInt(h_data_manual, n);
        end = std::chrono::high_resolution_clock::now();
        double manual_time = std::chrono::duration<double, std::milli>(end - start).count();

        // проверка результатов
        bool manual_correct = verify_sorted(h_data_manual, n);
        printf("Manual Radix Sort: %.3f ms\n", manual_time);

        // бенчмарк thrust sort
        start = std::chrono::high_resolution_clock::now();
        thrust::device_vector<int> d_thrust(h_data_thrust, h_data_thrust + n);
        thrust::sort(d_thrust.begin(), d_thrust.end());
        thrust::copy(d_thrust.begin(), d_thrust.end(), h_data_thrust);
        end = std::chrono::high_resolution_clock::now();
        double thrust_time = std::chrono::duration<double, std::milli>(end - start).count();

        // проверка результатов
        bool thrust_correct = verify_sorted(h_data_thrust, n);
        printf("Thrust Sort: %.3f ms\n", thrust_time);

        // бенчмарк std::sort (cpu)
        start = std::chrono::high_resolution_clock::now();
        std::sort(h_data_std, h_data_std + n);
        end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();

        // проверка результатов
        bool cpu_correct = verify_sorted(h_data_std, n);
        printf("CPU Sort: %.3f ms\n", cpu_time);

        // вычисление ускорения
        if (cpu_time > 0 && manual_time > 0)
        {
            printf("Manual Radix Speedup vs CPU: %.2fx\n", cpu_time / manual_time);
        }
        if (thrust_time > 0 && manual_time > 0)
        {
            printf("Manual Radix Speedup vs Thrust: %.2fx\n", thrust_time / manual_time);
        }
        if (radix_time > 0 && manual_time > 0)
        {
            printf("Manual Radix Speedup vs Thrust Radix: %.2fx\n", radix_time / manual_time);
        }

        delete[] h_data_radix;
        delete[] h_data_thrust;
        delete[] h_data_manual;
        delete[] h_data_std;
    }

    printf("\nTesting 64-bit integers:\n");
    int sizes64[] = {100000, 1000000, 10000000};
    int numSizes64 = sizeof(sizes64) / sizeof(sizes64[0]);

    for (int s = 0; s < numSizes64; s++)
    {
        int n64 = sizes64[s];
        printf("\n64-bit testing with array size: %d\n", n64);

        long long *h_data64_radix = new long long[n64];
        long long *h_data64_thrust = new long long[n64];
        long long *h_data64_manual = new long long[n64];
        long long *h_data64_std = new long long[n64];

        srand(99 + s);
        for (int i = 0; i < n64; i++)
        {
            h_data64_radix[i] = ((long long)rand() << 32) | rand();
            h_data64_radix[i] ^= (i & 1) ? 0x8000000000000000ULL : 0; // случайно инвертируем знаковый бит

            h_data64_thrust[i] = h_data64_radix[i];
            h_data64_manual[i] = h_data64_radix[i];
            h_data64_std[i] = h_data64_radix[i];
        }

        auto start = std::chrono::high_resolution_clock::now();
        radixSortLong(h_data64_radix, n64);
        auto end = std::chrono::high_resolution_clock::now();
        double radix64_time = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        manualRadixSortLong(h_data64_manual, n64);
        auto end_manual = std::chrono::high_resolution_clock::now();
        double manual64_time = std::chrono::duration<double, std::milli>(end_manual - start).count();

        start = std::chrono::high_resolution_clock::now();
        thrust::device_vector<long long> d_thrust64(h_data64_thrust, h_data64_thrust + n64);
        thrust::sort(d_thrust64.begin(), d_thrust64.end());
        thrust::copy(d_thrust64.begin(), d_thrust64.end(), h_data64_thrust);
        end = std::chrono::high_resolution_clock::now();
        double thrust64_time = std::chrono::duration<double, std::milli>(end - start).count();

        // cортировка на cpu для 64-битных чисел
        start = std::chrono::high_resolution_clock::now();
        std::sort(h_data64_std, h_data64_std + n64);
        end = std::chrono::high_resolution_clock::now();
        double cpu64_time = std::chrono::duration<double, std::milli>(end - start).count();

        bool radix64_correct = verify_sorted_long(h_data64_radix, n64);
        bool manual64_correct = verify_sorted_long(h_data64_manual, n64);
        bool thrust64_correct = verify_sorted_long(h_data64_thrust, n64);
        bool cpu64_correct = verify_sorted_long(h_data64_std, n64);

        printf("64-bit Thrust Radix Sort: %.3f ms\n", radix64_time);
        printf("64-bit Manual Radix Sort: %.3f ms\n", manual64_time);
        printf("64-bit Thrust Sort: %.3f ms\n", thrust64_time);
        printf("64-bit CPU Sort: %.3f ms\n", cpu64_time);

        // вычисление ускорения для 64-бит
        if (cpu64_time > 0 && manual64_time > 0)
        {
            printf("64-bit Manual Radix Speedup vs CPU: %.2fx\n", cpu64_time / manual64_time);
        }
        if (thrust64_time > 0 && manual64_time > 0)
        {
            printf("64-bit Manual Radix Speedup vs Thrust: %.2fx\n", thrust64_time / manual64_time);
        }
        if (radix64_time > 0 && manual64_time > 0)
        {
            printf("64-bit Manual Radix Speedup vs Thrust Radix: %.2fx\n", radix64_time / manual64_time);
        }

        delete[] h_data64_radix;
        delete[] h_data64_thrust;
        delete[] h_data64_manual;
        delete[] h_data64_std;
    }

    return 0;
}