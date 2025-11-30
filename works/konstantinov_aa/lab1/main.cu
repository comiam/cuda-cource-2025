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

__global__ void circle_kernel(char* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int dx = x - width / 2;;
    int dy = y - height / 2;
    int outer = radius * radius;
    int inner = (radius - 1) * (radius - 1);
    int dist2 = dx * dx + dy * dy;

    if (dist2 <= outer && dist2 >= inner) {
        output[idx] = '* ';
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
    const int radius = min(width, height) / 2 - 1;
    if (radius < 1) {
        std::cerr << "The field is too small to draw a circle.\n";
        return EXIT_FAILURE;
    }

    char* host = new char[size];
    memset(host, ' ', size);

    char* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, size));
    CUDA_CHECK(cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    circle_kernel<<<grid, block>>>(dev, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost));

    for (int y = 0; y < height; ++y) {
        printf("%.*s\n", width, host + y * width);
    }

    CUDA_CHECK(cudaFree(dev));
    delete[] host;

    return 0;
}