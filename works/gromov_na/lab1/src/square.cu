#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (expr);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                  \
        }                                                         \
    } while (0)

__global__ void draw_square(char *sq, int W, int H)
{
    // вычисление координат текущего потока в сетке
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // проверка на выход за границы изображения
    if (x >= W || y >= H)
        return;

    // вычисление линейного индекса в буфере
    int idx = y * W + x;

    // вычисление координат границ прямоугольника
    int left = 0;
    int right = W - 1;
    int top = 0;
    int bottom = H - 1;

    // инициализация символа пробелом
    char c = ' ';

    // проверка, находится ли точка на горизонтальной границе
    bool on_h = (y == top || y == bottom) && x >= left && x <= right;
    // проверка, находится ли точка на вертикальной границе
    bool on_v = (x == left || x == right) && y >= top && y <= bottom;

    // если точка на границе, меняем символ на '*'
    if (on_h || on_v)
        c = '*';

    // запись символа в буфер
    sq[idx] = c;
}

int main(int argc, char *argv[])
{
    int W = 0;
    int H = 0;

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <width> <height>\n";
        std::cerr << "Example: " << argv[0] << " 15 15\n";
        return EXIT_FAILURE;
    }

    W = atoi(argv[1]);
    H = atoi(argv[2]);

    if (W <= 0 || H <= 0)
    {
        std::cerr << "Invalid dimensions. Width and height must be positive integers.\n";
        return EXIT_FAILURE;
    }

    const int size = W * H;

    // выделение памяти на хосте
    std::vector<char> host(size, ' ');

    // выделение памяти на GPU
    char *dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, size));
    CUDA_CHECK(cudaMemset(dev, ' ', size)); // инициализация пробелами

    // настройка параметров запуска ядра
    dim3 block(32, 32);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    // запуск ядра
    draw_square<<<grid, block>>>(dev, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    // копирование результатов обратно на хост
    CUDA_CHECK(cudaMemcpy(host.data(), dev, size, cudaMemcpyDeviceToHost));

    // вывод результата
    for (int y = 0; y < H; ++y)
    {
        printf("%.*s\n", W, host.data() + y * W);
    }

    // освобождение ресурсов
    cudaFree(dev);

    return 0;
}