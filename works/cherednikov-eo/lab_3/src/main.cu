#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "error_check.cuh"

#define TILE 16

__global__ void sobel_filter(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char tile_s[TILE + 2][TILE + 2];

    int thread_x = threadIdx.x + 1;
    int thread_y = threadIdx.y + 1;
    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;

    //Загрузка центрального Тайла
    if (x < width && y < height) {
        tile_s[x][y] = input[y * width + x];
    } else {
        tile_s[x][y] = 0.0f
    }

    //Загрузка границ
    if (threadIdx.x.x == 0) {
        // Левый край
        int apron_x = max(0, x - 1);
        if (y < height) {
            tile_s[thread_y][0] = input[y * width + apron_x];
        } else {
            tile_s[thread_y][0] = 0.0f;
        }
    }

    if (threadIdx.x.x == TILE - 1 || x == width - 1) {
        // Правый край
        int apron_x = min(width - 1, x + 1);
        if (y < height && apron_x < width) {
            tile_s[thread_y][thread_x + 1] = input[y * width + apron_x];
        } else {
            tile_s[thread_y][thread_x + 1] = 0.0f;
        }
    }

    if (threadIdx.y == 0) {
        // Верхний край
        int apron_y = max(0, y - 1);
        if (x < width) {
            tile_s[0][thread_x] = input[apron_y * width + x];
        } else {
            tile_s[0][thread_x] = 0.0f;
        }
    }

    if (threadIdx.y == TILE - 1 || y == height - 1) {
        // Нижний край
        int apron_y = min(height - 1, y + 1);
        if (x < width && apron_y < height) {
            tile_s[thread_y + 1][thread_x] = input[apron_y * width + x];
        } else {
            tile_s[thread_y + 1][thread_x] = 0.0f;
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
            tile_s[0][thread_x + 1] = input[apron_y * width + apron_x];
        }
    }

    if (threadIdx.x == 0 && (threadIdx.y == TILE - 1 || y == height - 1)) {
        int apron_x = max(0, x - 1);
        int apron_y = min(height - 1, y + 1);
        if (apron_y < height) {
            tile_s[thread_y + 1][0] = input[apron_y * width + apron_x];
        }
    }

    if ((threadIdx.x == TILE - 1 || x == width - 1) &&
        (threadIdx.y == TILE - 1 || y == height - 1)) {
        int apron_x = min(width - 1, x + 1);
        int apron_y = min(height - 1, y + 1);
        if (apron_x < width && apron_y < height) {
            tile_s[thread_y + 1][thread_x + 1] = input[apron_y * width + apron_x];
        }
    }

    __syncthreads();

    // Высчитываем градиент Собеля
    if (x < width && y < height) {
        // Горизонтальный градиент Gx
        float gx = -1.0f * tile_s[thread_y - 1][thread_x - 1] + 1.0f * tile_s[thread_y - 1][thread_x + 1]
                   -2.0f * tile_s[thread_y][thread_x - 1]     + 2.0f * tile_s[thread_y][thread_x + 1]
                   -1.0f * tile_s[thread_y + 1][thread_x - 1] + 1.0f * tile_s[thread_y + 1][thread_x + 1];

        // Вертикальный градиент Gy
        float gy = -1.0f * tile_s[thread_y - 1][thread_x - 1] - 2.0f * tile_s[thread_y - 1][thread_x] - 1.0f * tile_s[thread_y - 1][thread_x + 1]
                   +1.0f * tile_s[thread_y + 1][thread_x - 1] + 2.0f * tile_s[thread_y + 1][thread_x] + 1.0f * tile_s[thread_y + 1][thread_x + 1];

        int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
        magnitude = min(255, max(0, magnitude));

        output[y * width + x] = (unsigned char)magnitude;
    }
}


int main() {

    return 0;
}
