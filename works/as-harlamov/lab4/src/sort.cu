#include "sort.h"
#include "utils.h"
#include <cstdint>

#define ZERO_BANK_CONFLICTS
#define MAX_BLOCK_SIZE 128
#define SHARED_MEMORY_BANKS 32
#define LOG_SHARED_MEMORY_BANKS 5

// Макрос для вычисления смещения, предотвращающего банковые конфликты
// Добавляем смещение к индексам, чтобы распределить обращения по разным банкам,
// что позволяет всем потокам обращаться к shared memory параллельно без задержек
#ifdef ZERO_BANK_CONFLICTS
#define BANK_CONFLICT_OFFSET(n) \
    (((n) >> LOG_SHARED_MEMORY_BANKS) + ((n) >> (2 * LOG_SHARED_MEMORY_BANKS)))
#else
#define BANK_CONFLICT_OFFSET(n) ((n) >> LOG_SHARED_MEMORY_BANKS)
#endif


// ============================================================================
// ЧАСТЬ 1: БАЗОВЫЕ ФУНКЦИИ ПРЕФИКСНЫХ СУММ
// ============================================================================

/**
 * Параллельное префиксное сканирование (prefix scan) с использованием алгоритма Blelloch
 * 
 * Префиксное сканирование вычисляет для каждого элемента сумму всех предыдущих элементов
 * Например, для [1, 2, 3, 4] результат: [0, 1, 3, 6]
 * 
 * Алгоритм Blelloch состоит из двух фаз:
 * 1. Up-sweep (восходящее сканирование): строим бинарное дерево сумм
 * 2. Down-sweep (нисходящее сканирование): распространяем суммы обратно к листьям
 */
__global__
void compute_prefix_scan(unsigned int* const device_output,
    const unsigned int* const device_input,
    unsigned int* const device_block_prefixes,
    const unsigned int input_length,
    const unsigned int shared_memory_size,
    const unsigned int elements_per_block) {
    extern __shared__ unsigned int shared_output[];

    int thread_id = threadIdx.x;
    int first_index = thread_id;
    int second_index = thread_id + blockDim.x;

    // Инициализируем shared memory нулями С УЧЕТОМ СМЕЩЕНИЯ для избежания банковых конфликтов
    shared_output[thread_id] = 0;
    shared_output[thread_id + blockDim.x] = 0;
    shared_output[thread_id + blockDim.x + BANK_CONFLICT_OFFSET(thread_id + blockDim.x)] = 0;

    __syncthreads();

    // Загружаем данные из глобальной памяти в shared memory
    unsigned int copy_index = elements_per_block * blockIdx.x + threadIdx.x;
    if (copy_index < input_length) {
        shared_output[first_index + BANK_CONFLICT_OFFSET(first_index)] = device_input[copy_index];
        if (copy_index + blockDim.x < input_length)
            shared_output[second_index + BANK_CONFLICT_OFFSET(second_index)] = device_input[copy_index + blockDim.x];
    }

    // ФАЗА 1: Up-sweep - строим бинарное дерево сумм снизу вверх
    // O(log n) вместо O(n)
    int stride = 1;
    for (int depth = elements_per_block >> 1; depth > 0; depth >>= 1) {
        __syncthreads();

        if (thread_id < depth) {
            int first_idx = stride * ((thread_id << 1) + 1) - 1;
            int second_idx = stride * ((thread_id << 1) + 2) - 1;
            first_idx += BANK_CONFLICT_OFFSET(first_idx);
            second_idx += BANK_CONFLICT_OFFSET(second_idx);

            shared_output[second_idx] += shared_output[first_idx];
        }
        stride <<= 1;
    }

    // Сохраняем сумму всего блока для последующего накопления между блоками
    // Обнуляем последний элемент для корректной работы down-sweep фазы
    if (thread_id == 0) {
        device_block_prefixes[blockIdx.x] = shared_output[elements_per_block - 1 + BANK_CONFLICT_OFFSET(elements_per_block - 1)];
        shared_output[elements_per_block - 1 + BANK_CONFLICT_OFFSET(elements_per_block - 1)] = 0;
    }

    // ФАЗА 2: Down-sweep - распространяем суммы обратно к листьям дерева
    // Теперь каждый элемент получает сумму всех предыдущих элементов
    for (int depth = 1; depth < elements_per_block; depth <<= 1) {
        stride >>= 1;
        __syncthreads();

        if (thread_id < depth) {
            int first_idx = stride * ((thread_id << 1) + 1) - 1;
            int second_idx = stride * ((thread_id << 1) + 2) - 1;
            first_idx += BANK_CONFLICT_OFFSET(first_idx);
            second_idx += BANK_CONFLICT_OFFSET(second_idx);

            unsigned int swap_temp = shared_output[first_idx];
            shared_output[first_idx] = shared_output[second_idx];
            shared_output[second_idx] += swap_temp;
        }
    }
    __syncthreads();

    // Записываем результаты обратно в глобальную память
    if (copy_index < input_length) {
        device_output[copy_index] = shared_output[first_index + BANK_CONFLICT_OFFSET(first_index)];
        if (copy_index + blockDim.x < input_length)
            device_output[copy_index + blockDim.x] = shared_output[second_index + BANK_CONFLICT_OFFSET(second_index)];
    }
}

/**
 * Накопление префиксов блоков
 * 
 * После вычисления префиксных сумм внутри каждого блока добавляем
 * к результатам каждого блока сумму всех предыдущих блоков. Это позволяет
 * нам получить глобальные префиксные суммы для всего массива
 */
__global__
void accumulate_block_prefixes(unsigned int* const device_output,
    const unsigned int* const device_input,
    unsigned int* const device_block_prefixes,
    const size_t total_elements) {
    unsigned int block_prefix_value = device_block_prefixes[blockIdx.x];

    unsigned int copy_index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (copy_index < total_elements) {
        device_output[copy_index] = device_input[copy_index] + block_prefix_value;
        if (copy_index + blockDim.x < total_elements)
            device_output[copy_index + blockDim.x] = device_input[copy_index + blockDim.x] + block_prefix_value;
    }
}

/**
 * Параллельное префиксное суммирование
 * 
 * Для больших массивов, которые не помещаются в один блок потоков, используем
 * двухуровневую стратегию:
 * 1. Вычисляем префиксные суммы внутри каждого блока
 * 2. Рекурсивно вычисляем префиксные суммы для сумм блоков
 * 3. Добавляем префиксы блоков к результатам внутри блоков
 */
void parallel_prefix_sum(unsigned int* const device_output,
    const unsigned int* const device_input,
    const size_t element_count) {
    CUDA_CHECK(cudaMemset(device_output, 0, element_count * sizeof(unsigned int)));

    // Каждый поток обрабатывает 2 элемента
    unsigned int block_size = MAX_BLOCK_SIZE / 2;
    unsigned int max_elements_per_block = 2 * block_size;

    unsigned int grid_size = element_count / max_elements_per_block;
    if (element_count % max_elements_per_block != 0)
        grid_size += 1;

    // Вычисляем размер shared memory с учетом смещений для избежания банковых конфликтов
    // Максимальный индекс = max_elements_per_block - 1, но для безопасности используем max_elements_per_block
    unsigned int max_offset = BANK_CONFLICT_OFFSET(max_elements_per_block);
    unsigned int shared_memory_size = max_elements_per_block + max_offset;

    // Выделяем память для хранения сумм каждого блока
    unsigned int* device_block_prefixes;
    CUDA_CHECK(cudaMalloc(&device_block_prefixes, sizeof(unsigned int) * grid_size));
    CUDA_CHECK(cudaMemset(device_block_prefixes, 0, sizeof(unsigned int) * grid_size));

    // ШАГ 1: Вычисляем префиксные суммы внутри каждого блока
    // Результаты сохраняются в device_output, суммы блоков - в device_block_prefixes
    compute_prefix_scan<<<grid_size, block_size, sizeof(unsigned int) * shared_memory_size>>>(
        device_output, device_input, device_block_prefixes, element_count, shared_memory_size, max_elements_per_block);

    // ШАГ 2: Вычисляем префиксные суммы для сумм блоков
    // Если блоков немного - используем упрощенный однопроходный алгоритм
    // Если блоков много - рекурсивно вызываем эту же функцию
    if (grid_size <= max_elements_per_block) {
        unsigned int* device_dummy_prefixes;
        CUDA_CHECK(cudaMalloc(&device_dummy_prefixes, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(device_dummy_prefixes, 0, sizeof(unsigned int)));
        compute_prefix_scan<<<1, block_size, sizeof(unsigned int) * shared_memory_size>>>(
            device_block_prefixes, device_block_prefixes, device_dummy_prefixes, grid_size, shared_memory_size,
            max_elements_per_block);
        CUDA_CHECK(cudaFree(device_dummy_prefixes));
    } else {
        unsigned int* device_input_block_prefixes;
        CUDA_CHECK(cudaMalloc(&device_input_block_prefixes, sizeof(unsigned int) * grid_size));
        CUDA_CHECK(cudaMemcpy(device_input_block_prefixes, device_block_prefixes, sizeof(unsigned int) * grid_size, cudaMemcpyDeviceToDevice));
        parallel_prefix_sum(device_block_prefixes, device_input_block_prefixes, grid_size);
        CUDA_CHECK(cudaFree(device_input_block_prefixes));
    }

    // ШАГ 3: Добавляем префиксы блоков к результатам внутри каждого блока
    // Теперь каждый элемент содержит глобальную префиксную сумму
    accumulate_block_prefixes<<<grid_size, block_size>>>(device_output, device_output, device_block_prefixes,
                                               element_count);

    CUDA_CHECK(cudaFree(device_block_prefixes));
}

// ============================================================================
// ЧАСТЬ 2: ФУНКЦИИ RADIX SORT
// ============================================================================

/**
 * Инверсия знакового бита для поддержки отрицательных чисел в Radix Sort
 */
template<typename T>
__global__ void flip_sign_bit(T* device_data, unsigned int input_length) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_length) {
        // Для 32-битных: инвертируем бит 31, для 64-битных: бит 63
        if (sizeof(T) == 4) {
            uint32_t* data_uint = reinterpret_cast<uint32_t*>(device_data);
            data_uint[idx] ^= 0x80000000UL;
        } else if (sizeof(T) == 8) {
            uint64_t* data_uint = reinterpret_cast<uint64_t*>(device_data);
            data_uint[idx] ^= 0x8000000000000000ULL;
        }
    }
}

/**
 * Локальный шаг Radix Sort для одного блока данных
 * 
 * Radix Sort сортирует числа по разрядам (битам), начиная с младших разрядов
 * На каждом шаге мы извлекаем 2 бита (4 возможных значения: 0, 1, 2, 3),
 * группируем элементы по этим значениям и переставляем их в правильном порядке
 * 
 * Алгоритм для одного блока:
 * 1. Загружаем данные блока в shared memory
 * 2. Для каждой цифры (0-3): создаем маску, вычисляем префиксные суммы маски
 * 3. Вычисляем позиции элементов на основе префиксных сумм
 * 4. Переставляем элементы в отсортированном порядке внутри блока
 */
template<typename T>
__global__ void local_radix_sort_step(T* device_output_sorted,
    unsigned int* device_prefix_sums,
    unsigned int* device_block_sums,
    unsigned int bit_shift_amount,
    T* device_input,
    unsigned int input_length,
    unsigned int max_elements_per_block) {
    extern __shared__ unsigned int shared_memory[];
    T* shared_data = (T*)shared_memory;
    // Вычисляем смещение для масок и префиксных сумм
    unsigned int shared_data_size_uint = (max_elements_per_block * sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    unsigned int shared_mask_length = max_elements_per_block + 1;
    unsigned int* shared_mask_output = &shared_memory[shared_data_size_uint];  // Маски для каждой цифры
    unsigned int* shared_merged_scan = &shared_mask_output[shared_mask_length];  // Префиксные суммы
    unsigned int* shared_mask_sums = &shared_merged_scan[max_elements_per_block];  // Суммы по цифрам
    unsigned int* shared_scan_mask_sums = &shared_mask_sums[4];  // Префиксные суммы сумм

    unsigned int thread_id = threadIdx.x;

    // Загружаем данные блока в shared memory
    unsigned int copy_index = max_elements_per_block * blockIdx.x + thread_id;
    if (copy_index < input_length) {
        shared_data[thread_id] = device_input[copy_index];
    } else {
        shared_data[thread_id] = 0;
    }

    __syncthreads();

    // Извлекаем 2 бита (цифры от 0 до 3) из текущего разряда числа
    T thread_data = shared_data[thread_id];
    // Преобразуем в беззнаковый тип для корректной работы с битами
    uint32_t thread_data_uint32 = 0;
    uint64_t thread_data_uint64 = 0;
    if (sizeof(T) == 4) {
        thread_data_uint32 = (uint32_t)thread_data;
    } else if (sizeof(T) == 8) {
        thread_data_uint64 = (uint64_t)thread_data;
    }
    unsigned int extracted_bits = (sizeof(T) == 4) ? 
        ((thread_data_uint32 >> bit_shift_amount) & 3) : 
        ((thread_data_uint64 >> bit_shift_amount) & 3);

    // Обрабатываем каждую из 4 возможных цифр (0, 1, 2, 3)
    // Для каждой цифры вычисляем, сколько элементов имеют эту цифру и их позиции
    for (unsigned int digit_value = 0; digit_value < 4; digit_value++) {
        // Инициализируем маску для текущей цифры
        shared_mask_output[thread_id] = 0;
        if (thread_id == 0)
            shared_mask_output[shared_mask_length - 1] = 0;

        __syncthreads();

        // Создаем маску: 1 если элемент имеет текущую цифру, 0 иначе
        bool matches_digit = false;
        if (copy_index < input_length) {
            matches_digit = extracted_bits == digit_value;
            shared_mask_output[thread_id] = matches_digit;
        }
        __syncthreads();

        // Параллельно вычисляем префиксные суммы маски
        // Каждый поток суммирует свое значение с предыдущим на расстоянии 2^step
        // Получается O(log n) вместо O(n)
        int partner_thread = 0;
        unsigned int partial_sum = 0;
        unsigned int iteration_count = (unsigned int) log2f(max_elements_per_block);
        for (unsigned int step = 0; step < iteration_count; step++) {
            partner_thread = thread_id - (1 << step);
            if (partner_thread >= 0) {
                partial_sum = shared_mask_output[thread_id] + shared_mask_output[partner_thread];
            } else {
                partial_sum = shared_mask_output[thread_id];
            }
            __syncthreads();
            shared_mask_output[thread_id] = partial_sum;
            __syncthreads();
        }

        // Сдвигаем префиксные суммы на 1 позицию вправо для получения правильных индексов
        unsigned int copy_value = shared_mask_output[thread_id];
        __syncthreads();
        shared_mask_output[thread_id + 1] = copy_value;
        __syncthreads();

        // Сохраняем общее количество элементов с текущей цифрой в блоке
        // Это нужно  для последующей глобальной перестановки между блоками
        if (thread_id == 0) {
            shared_mask_output[0] = 0;
            unsigned int block_total = shared_mask_output[shared_mask_length - 1];
            shared_mask_sums[digit_value] = block_total;
            device_block_sums[digit_value * gridDim.x + blockIdx.x] = block_total;
        }
        __syncthreads();

        // Сохраняем префиксную сумму для элементов с текущей цифрой
        if (matches_digit && (copy_index < input_length)) {
            shared_merged_scan[thread_id] = shared_mask_output[thread_id];
        }

        __syncthreads();
    }

    // Вычисляем префиксные суммы для сумм по цифрам
    // Это позволяет определить глобальную позицию каждого элемента внутри блока
    if (thread_id == 0) {
        unsigned int running_total = 0;
        for (unsigned int digit_idx = 0; digit_idx < 4; digit_idx++) {
            shared_scan_mask_sums[digit_idx] = running_total;
            running_total += shared_mask_sums[digit_idx];
        }
    }

    __syncthreads();

    // Переставляем элементы в отсортированном порядке внутри блока
    // Каждый элемент получает новую позицию на основе своей цифры и префиксных сумм
    if (copy_index < input_length) {
        unsigned int thread_prefix_sum = shared_merged_scan[thread_id];
        // Новая позиция = локальная префиксная сумма + смещение для текущей цифры
        unsigned int new_position = thread_prefix_sum + shared_scan_mask_sums[extracted_bits];

        __syncthreads();

        // Записываем элемент в новую позицию в shared memory
        shared_data[new_position] = thread_data;
        shared_merged_scan[new_position] = thread_prefix_sum;

        __syncthreads();

        // Сохраняем результаты в глобальную память
        device_prefix_sums[copy_index] = shared_merged_scan[thread_id];
        device_output_sorted[copy_index] = shared_data[thread_id];
    }
}

/**
 * Глобальная перестановка данных между блоками после локальной сортировки
 * 
 * После локальной сортировки внутри каждого блока элементы отсортированы локально,
 * но нам необходимо переставить их глобально, чтобы все элементы с цифрой 0 были перед
 * элементами с цифрой 1, те - перед элементами с цифрой 2, и т.д.
 * 
 * Алгоритм:
 * 1. Для каждого элемента вычисляем его глобальную позицию
 * 2. Глобальная позиция = смещение для цифры элемента + локальная позиция в блоке
 * 3. Записываем элемент в глобальную позицию
 */
template<typename T>
__global__ void global_data_rearrangement(T* device_output,
    T* device_input,
    unsigned int* device_scan_block_sums,
    unsigned int* device_prefix_sums,
    unsigned int bit_shift_amount,
    unsigned int input_length,
    unsigned int max_elements_per_block) {
    unsigned int thread_id = threadIdx.x;
    unsigned int copy_index = max_elements_per_block * blockIdx.x + thread_id;

    if (copy_index < input_length) {
        T thread_data = device_input[copy_index];
        uint32_t thread_data_uint32 = 0;
        uint64_t thread_data_uint64 = 0;
        if (sizeof(T) == 4) {
            thread_data_uint32 = (uint32_t)thread_data;
        } else if (sizeof(T) == 8) {
            thread_data_uint64 = (uint64_t)thread_data;
        }
        unsigned int extracted_bits = (sizeof(T) == 4) ? 
            ((thread_data_uint32 >> bit_shift_amount) & 3) : 
            ((thread_data_uint64 >> bit_shift_amount) & 3);
        
        // Получаем локальную позицию элемента в блоке (из предыдущего шага)
        unsigned int thread_prefix_sum = device_prefix_sums[copy_index];
        
        // Вычисляем глобальную позицию элемента
        // device_scan_block_sums содержит смещения для каждой цифры в каждом блоке
        // thread_prefix_sum - локальная позиция внутри блока
        unsigned int global_position =
            device_scan_block_sums[extracted_bits * gridDim.x + blockIdx.x] +
            thread_prefix_sum;
        __syncthreads();
        
        // Записываем элемент в глобальную позицию
        device_output[global_position] = thread_data;
    }
}

/**
 * Основная функция Radix Sort для сортировки массива знаковых целых чисел.
 * 
 * Алгоритм состоит из следующих шагов (повторяются для каждого разряда):
 * 1. Инвертируем знаковый бит для поддержки отрицательных чисел
 * 2. Выполняем локальную сортировку внутри каждого блока (local_radix_sort_step)
 * 3. Вычисляем префиксные суммы для сумм блоков (parallel_prefix_sum)
 * 4. Выполняем глобальную перестановку элементов между блоками (global_data_rearrangement)
 * 5. Инвертируем знаковый бит обратно для восстановления исходных значений
 */
template<typename T>
void radix_sort(T* const device_output,
    T* const device_input,
    unsigned int input_length) {

    unsigned int block_size = MAX_BLOCK_SIZE;
    unsigned int max_elements_per_block = block_size;
    unsigned int grid_size = input_length / max_elements_per_block;
    if (input_length % max_elements_per_block != 0)
        grid_size += 1;

    // Эти суммы используем для определения позиций элементов при перестановке
    unsigned int* device_prefix_sums;
    unsigned int prefix_sums_length = input_length;
    CUDA_CHECK(cudaMalloc(&device_prefix_sums, sizeof(unsigned int) * prefix_sums_length));
    CUDA_CHECK(cudaMemset(device_prefix_sums, 0, sizeof(unsigned int) * prefix_sums_length));

    // Здесь храним суммы элементов по цифрам в каждом банке
    // [цифра 0 блок 0, цифра 0 блок 1, ..., цифра 1 блок 0, ...]
    unsigned int* device_block_sums;
    unsigned int block_sums_length = 4 * grid_size;
    CUDA_CHECK(cudaMalloc(&device_block_sums, sizeof(unsigned int) * block_sums_length));
    CUDA_CHECK(cudaMemset(device_block_sums, 0, sizeof(unsigned int) * block_sums_length));

    // Префиксные суммы сумм блоков
    unsigned int* device_scan_block_sums;
    CUDA_CHECK(cudaMalloc(&device_scan_block_sums, sizeof(unsigned int) * block_sums_length));
    CUDA_CHECK(cudaMemset(device_scan_block_sums, 0, sizeof(unsigned int) * block_sums_length));

    
    unsigned int shared_data_size = (max_elements_per_block * sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    unsigned int shared_mask_length = max_elements_per_block + 1;
    unsigned int shared_merged_scan_length = max_elements_per_block;
    unsigned int shared_mask_sums_length = 4;
    unsigned int shared_scan_mask_sums_length = 4;
    unsigned int shared_memory_size = (shared_data_size + shared_mask_length +
                             shared_merged_scan_length + shared_mask_sums_length +
                             shared_scan_mask_sums_length) * sizeof(unsigned int);
    
    // ШАГ 0: Инвертируем знаковый бит для поддержки отрицательных чисел
    unsigned int flip_grid_size = (input_length + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    flip_sign_bit<T><<<flip_grid_size, MAX_BLOCK_SIZE>>>(device_input, input_length);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Основной цикл Radix Sort: обрабатываем каждый разряд (2 бита за раз)
    // Для 32-битных чисел: 16 итераций (биты 0-1, 2-3, 4-5, ..., 30-31)
    // Для 64-битных чисел: 32 итерации (биты 0-1, 2-3, 4-5, ..., 62-63)
    unsigned int max_bit_shift = (sizeof(T) * 8) - 2;  // sizeof(T)*8 - количество бит, -2 для обработки по 2 бита
    for (unsigned int bit_shift = 0; bit_shift <= max_bit_shift; bit_shift += 2) {
        // ШАГ 1: Выполняем локальную сортировку внутри каждого блока
        // Элементы переставляются в отсортированном порядке внутри блока
        local_radix_sort_step<T><<<grid_size, block_size, shared_memory_size>>>(
            device_output, device_prefix_sums, device_block_sums, bit_shift, device_input, input_length,
            max_elements_per_block);

        // ШАГ 2: Вычисляем префиксные суммы для сумм блоков
        // Это позволяет определить глобальные смещения для каждой цифры
        parallel_prefix_sum(device_scan_block_sums, device_block_sums, block_sums_length);

        // ШАГ 3: Выполняем глобальную перестановку элементов между блоками
        // Элементы записываются в правильные глобальные позиции
        global_data_rearrangement<T><<<grid_size, block_size>>>(
            device_input, device_output, device_scan_block_sums, device_prefix_sums, bit_shift,
            input_length, max_elements_per_block);
    }
    
    CUDA_CHECK(cudaMemcpy(device_output, device_input, sizeof(T) * input_length, cudaMemcpyDeviceToDevice));
    
    // ШАГ 4: Инвертируем знаковый бит обратно для восстановления исходных значений
    flip_sign_bit<T><<<flip_grid_size, MAX_BLOCK_SIZE>>>(device_output, input_length);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(device_scan_block_sums));
    CUDA_CHECK(cudaFree(device_block_sums));
    CUDA_CHECK(cudaFree(device_prefix_sums));
}

template void radix_sort<int>(int* const device_output, int* const device_input, unsigned int input_length);
template void radix_sort<int64_t>(int64_t* const device_output, int64_t* const device_input, unsigned int input_length);