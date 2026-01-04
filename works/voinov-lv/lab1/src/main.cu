#include <stdio.h>
#include <cuda_runtime.h>

__global__ void drawCircle(char* pixels, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float dx = x - width / 2;
        float dy = (y - height / 2) * 2;
        float dist = sqrtf(dx * dx + dy * dy);
        
        if (fabsf(dist - radius) < 1) {
            pixels[y * width + x] = '#';
        } else {
            pixels[y * width + x] = ' ';
        }
    }
}

int main() {
    const int width = 40, height = 20, radius = 15;
    char h_pixels[width * height];
    char* d_pixels;
    
    cudaMalloc(&d_pixels, width * height * sizeof(char)); // выделение памяти
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    drawCircle<<<gridSize, blockSize>>>(d_pixels, width, height, radius); // запуск ядра

    cudaDeviceSynchronize();
    
    cudaMemcpy(h_pixels, d_pixels, width * height, cudaMemcpyDeviceToHost); // копирование на cpu
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%c", h_pixels[y * width + x]);
        }
        printf("\n");
    }
    
    cudaFree(d_pixels);
    
    return 0;
}