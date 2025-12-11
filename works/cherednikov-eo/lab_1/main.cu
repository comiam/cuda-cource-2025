#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "error_check.cuh"


//__global__ void hello_kernel() {
    // device-side printf разрешён: можно логировать поток/блок.
//   printf("Block(%d,%d,%d) Thread(%d,%d,%d)\n",
//           blockIdx.x, blockIdx.y, blockIdx.z,
//           threadIdx.x, threadIdx.y, threadIdx.z);
//}


__global__ void draw_square(char* canvas, int w, int h, int margin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;

    // Условие: принадлежит ли пиксель границе квадрата
    bool inside = (x >= margin && x < w - margin && y >= margin && y < h - margin);
    bool border = (x == margin || x == w - margin - 1 || y == margin || y == h - margin - 1);

    if (inside && border)
        canvas[idx] = '*';
    else
        canvas[idx] = ' ';
}


int main() {
    int width = 28;
    int height = 28;
    int margin = 2;   // толщина отступа от края

    size_t size = width * height;
    char* h_canvas = (char*)malloc(size);
    char* d_canvas;

    cudaMalloc(&d_canvas, size);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    draw_square<<<grid, block>>>(d_canvas, width, height, margin);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(h_canvas, d_canvas, size, cudaMemcpyDeviceToHost);

    // Печать квадрата
    for (int y = 0; y < height; y++) {
        fwrite(&h_canvas[y * width], 1, width, stdout);
        putchar('\n');
    }

    cudaFree(d_canvas);
    free(h_canvas);
    return 0;
}
