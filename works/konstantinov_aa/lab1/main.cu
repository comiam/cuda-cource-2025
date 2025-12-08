#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                    \
        cudaError_t err = (expr);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

// Ядро CUDA для рисования круга на GPU
__global__ void circle_kernel(char* output, int width, int height, int radius) {
    // Вычисление глобальных координат потока
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Проверка выхода за границы массива
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Вычисление смещения относительно центра
    int dx = x - width / 2;
    int dy = y - height / 2;
    
    // Определение границ круга (в квадрате, чтобы избежать sqrt)
    int outer = radius * radius;
    int inner = (radius - 1) * (radius - 1);
    int dist2 = dx * dx + dy * dy;

    // Определение, является ли пиксель частью контура круга
    if (dist2 <= outer && dist2 >= inner) {
        output[idx] = '*';
    } else {
        output[idx] = ' ';
    }
}

int main() {
    int width = 0;
    int height = 0;

    std::cout << "Enter the field width and height (e.g., 60 30): ";
    if (!(std::cin >> width >> height) || width <= 0 || height <= 0) {
        std::cerr << "Invalid dimensions.\n";
        return EXIT_FAILURE;
    }

    const int size = width * height;
    const int radius = std::min(width, height) / 2 - 1;
    if (radius < 1) {
        std::cerr << "The field is too small to draw a circle.\n";
        return EXIT_FAILURE;
    }

    // Выделение памяти на хосте
    char* host = new char[size];
    memset(host, ' ', size);

    // Выделение памяти на устройстве (GPU)
    char* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, size));
    CUDA_CHECK(cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice));

    // Настройка параметров запуска ядра
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Запуск ядра
    circle_kernel<<<grid, block>>>(dev, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Копирование результатов обратно на хост
    CUDA_CHECK(cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost));

    // Вывод результата
    for (int y = 0; y < height; ++y) {
        printf("%.*s\n", width, host + y * width);
    }

    // Освобождение ресурсов
    CUDA_CHECK(cudaFree(dev));
    delete[] host;

    return 0;
}