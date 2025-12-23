#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define TILE_SIZE 16                     
#define BLOCK_SIZE (TILE_SIZE + 2)        


__constant__ int SOBEL_X[3][3] = {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
};

__constant__ int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

// Фильтр Собеля с Tiling оптимизацией

__global__ void sobel_kernel_tiled(const unsigned char* input, unsigned char* output, int width, int height) {
    
    // Shared memory для тайла + границы для свёртки 3x3
    __shared__ unsigned char tile[BLOCK_SIZE][BLOCK_SIZE];
    
    // Глобальные координаты левого верхнего угла тайла (включая границы)
    int tileStartX = blockIdx.x * TILE_SIZE - 1;
    int tileStartY = blockIdx.y * TILE_SIZE - 1;
    
    // Используем stride loop для загрузки 18x18=324 пикселей силами 16x16=256 потоков (некоторые потоки загрузят по 2 пикселя)
    int linearIdx = threadIdx.y * TILE_SIZE + threadIdx.x;  // Линейный индекс потока (0..255)
    int totalPixels = BLOCK_SIZE * BLOCK_SIZE;              // Всего пикселей в тайле (324)
    int numThreads = TILE_SIZE * TILE_SIZE;                 // Количество потоков (256)
    
    for (int i = linearIdx; i < totalPixels; i += numThreads) {
        // Преобразуем линейный индекс в 2D координаты внутри тайла
        int localY = i / BLOCK_SIZE;
        int localX = i % BLOCK_SIZE;
        
        // Глобальные координаты пикселя
        int globalX = tileStartX + localX;
        int globalY = tileStartY + localY;
        
        // Загружаем пиксель (или 0, если за границей изображения)
        if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
            tile[localY][localX] = input[globalY * width + globalX];
        } else {
            tile[localY][localX] = 0;
        }
    }
    __syncthreads();
    
   
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Проверяем, что пиксель внутри изображения
    if (x < width && y < height) {
        // Локальные координаты в tile (сдвиг +1 из-за границ)
        int tx = threadIdx.x + 1;
        int ty = threadIdx.y + 1;
        
        float gx = 0.0f;
        float gy = 0.0f;
        
       
        #pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            #pragma unroll
            for (int dx = -1; dx <= 1; dx++) {
                unsigned char pixel = tile[ty + dy][tx + dx];
                int ky = dy + 1;
                int kx = dx + 1;
                gx += pixel * SOBEL_X[ky][kx];
                gy += pixel * SOBEL_Y[ky][kx];
            }
        }
        
        // Магнитуда градиента
        float magnitude = abs(gx) + abs(gy);
        
        // Записываем результат (ограничиваем до 255)
        output[y * width + x] = (unsigned char)(magnitude > 255.0f ? 255.0f : magnitude);
    }
}

// Wrapper функция для вызова из C++
extern "C" int run_sobel_wrapper(const unsigned char* h_input, unsigned char* h_output, int width, int height) {
    unsigned char *d_input, *d_output;
    size_t imgSize = width * height * sizeof(unsigned char);
    cudaError_t err;

    err = cudaMalloc((void**)&d_input, imgSize);
    if (err != cudaSuccess) return 1;

    err = cudaMalloc((void**)&d_output, imgSize);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        return 1;
    }

    err = cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
        
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, 
                  (height + TILE_SIZE - 1) / TILE_SIZE);

    sobel_kernel_tiled<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        return 2;
    }

    err = cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }
   
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
