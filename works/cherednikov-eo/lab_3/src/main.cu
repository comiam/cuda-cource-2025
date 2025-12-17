#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../include/error_check.cuh"
#include "../include/image_io.cuh"

#define TILE 16

// Оператор Собеля для обнаружения границ
__global__ void sobel_filter(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char tile_s[TILE + 2][TILE + 2];

    // Локальные координаты в тайле (1-indexed для учета границ)
    int thread_x = threadIdx.x + 1;
    int thread_y = threadIdx.y + 1;
    
    // Глобальные координаты в изображении
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Загрузка центрального тайла
    if (x < width && y < height) {
        tile_s[thread_y][thread_x] = input[y * width + x];
    } else {
        tile_s[thread_y][thread_x] = 0;
    }

    // Загрузка левой границы
    if (threadIdx.x == 0) {
        int apron_x = max(0, x - 1);
        if (y < height) {
            tile_s[thread_y][0] = input[y * width + apron_x];
        } else {
            tile_s[thread_y][0] = 0;
        }
    }

    // Загрузка правой границы
    if (threadIdx.x == TILE - 1 || x == width - 1) {
        int apron_x = min(width - 1, x + 1);
        if (y < height && apron_x < width) {
            tile_s[thread_y][TILE + 1] = input[y * width + apron_x];
        } else {
            tile_s[thread_y][TILE + 1] = 0;
        }
    }

    // Загрузка верхней границы
    if (threadIdx.y == 0) {
        int apron_y = max(0, y - 1);
        if (x < width) {
            tile_s[0][thread_x] = input[apron_y * width + x];
        } else {
            tile_s[0][thread_x] = 0;
        }
    }

    // Загрузка нижней границы
    if (threadIdx.y == TILE - 1 || y == height - 1) {
        int apron_y = min(height - 1, y + 1);
        if (x < width && apron_y < height) {
            tile_s[TILE + 1][thread_x] = input[apron_y * width + x];
        } else {
            tile_s[TILE + 1][thread_x] = 0;
        }
    }

    // Загрузка углов
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int apron_x = max(0, x - 1);
        int apron_y = max(0, y - 1);
        tile_s[0][0] = input[apron_y * width + apron_x];
    }

    if ((threadIdx.x == TILE - 1 || x == width - 1) && threadIdx.y == 0) {
        int apron_x = min(width - 1, x + 1);
        int apron_y = max(0, y - 1);
        if (apron_x < width) {
            tile_s[0][TILE + 1] = input[apron_y * width + apron_x];
        } else {
            tile_s[0][TILE + 1] = 0;
        }
    }

    if (threadIdx.x == 0 && (threadIdx.y == TILE - 1 || y == height - 1)) {
        int apron_x = max(0, x - 1);
        int apron_y = min(height - 1, y + 1);
        if (apron_y < height) {
            tile_s[TILE + 1][0] = input[apron_y * width + apron_x];
        } else {
            tile_s[TILE + 1][0] = 0;
        }
    }

    if ((threadIdx.x == TILE - 1 || x == width - 1) &&
        (threadIdx.y == TILE - 1 || y == height - 1)) {
        int apron_x = min(width - 1, x + 1);
        int apron_y = min(height - 1, y + 1);
        if (apron_x < width && apron_y < height) {
            tile_s[TILE + 1][TILE + 1] = input[apron_y * width + apron_x];
        } else {
            tile_s[TILE + 1][TILE + 1] = 0;
        }
    }

    __syncthreads();

    // Вычисление градиента Собеля
    if (x < width && y < height) {
        // Горизонтальный градиент Gx
        float gx = -1.0f * tile_s[thread_y - 1][thread_x - 1] + 1.0f * tile_s[thread_y - 1][thread_x + 1]
                   -2.0f * tile_s[thread_y][thread_x - 1]     + 2.0f * tile_s[thread_y][thread_x + 1]
                   -1.0f * tile_s[thread_y + 1][thread_x - 1] + 1.0f * tile_s[thread_y + 1][thread_x + 1];

        // Вертикальный градиент Gy
        float gy = -1.0f * tile_s[thread_y - 1][thread_x - 1] - 2.0f * tile_s[thread_y - 1][thread_x] - 1.0f * tile_s[thread_y - 1][thread_x + 1]
                   +1.0f * tile_s[thread_y + 1][thread_x - 1] + 2.0f * tile_s[thread_y + 1][thread_x] + 1.0f * tile_s[thread_y + 1][thread_x + 1];

        // Вычисление магнитуды градиента
        float magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = fminf(255.0f, fmaxf(0.0f, magnitude));

        output[y * width + x] = (unsigned char)magnitude;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file.pgm> <output_file.pgm>\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    // Загрузка изображения
    unsigned char* host_input = nullptr;
    int width, height;
    
    printf("Loading image: %s\n", input_file);
    if (load_pgm(input_file, &host_input, &width, &height) != 0) {
        return 1;
    }
    printf("Image size: %d x %d\n", width, height);


    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    size_t image_size = width * height * sizeof(unsigned char);

    cudaMalloc((void**)&d_input, image_size);
    cudaMalloc((void**)&d_output, image_size);


    cudaMemcpy(d_input, host_input, image_size, cudaMemcpyHostToDevice);
    dim3 block_size(TILE, TILE);
    dim3 grid_size((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    printf("Launching kernel: grid(%d, %d), block(%d, %d)\n", 
           grid_size.x, grid_size.y, block_size.x, block_size.y);

    // Создание событий для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    sobel_filter<<<grid_size, block_size>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Измерение времени выполнения
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU execution time: %.3f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Копирование результата обратно на CPU
    unsigned char* host_output = (unsigned char*)malloc(image_size);
    if (!host_output) {
        fprintf(stderr, "Error: failed to allocate memory for result\n");
        cudaFree(d_input);
        cudaFree(d_output);
        free(host_input);
        return 1;
    }

    cudaMemcpy(host_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // Save result
    printf("Saving result: %s\n", output_file);
    if (save_pgm(output_file, host_output, width, height) != 0) {
        free(host_output);
        cudaFree(d_input);
        cudaFree(d_output);
        free(host_input);
        return 1;
    }

    // Освобождение памяти
    free(host_input);
    free(host_output);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Processing completed successfully!\n");
    return 0;
}
