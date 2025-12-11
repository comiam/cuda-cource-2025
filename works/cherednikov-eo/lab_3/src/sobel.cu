//
// Sobel operator CUDA implementation
//

#include "../headers/sobel.cuh"
#include <cmath>

// Маски оператора Собеля
// Gx = [-1  0  1]    Gy = [-1 -2 -1]
//      [-2  0  2]         [ 0  0  0]
//      [-1  0  1]         [ 1  2  1]

__device__ int sobelGx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__device__ int sobelGy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__global__ void sobelKernel(unsigned char* input, unsigned char* output, 
                            int width, int height) {
    // Вычисляем индекс потока
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем границы (игнорируем края изображения)
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) {
            output[y * width + x] = 0;
        }
        return;
    }

    // Применяем оператор Собеля
    int gx = 0, gy = 0;
    
    // Проходим по 3x3 окну вокруг пикселя
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int idx = (y + ky) * width + (x + kx);
            int pixel = input[idx];
            int maskIdx = (ky + 1) * 3 + (kx + 1);
            
            gx += pixel * sobelGx[maskIdx];
            gy += pixel * sobelGy[maskIdx];
        }
    }

    // Вычисляем магнитуду градиента
    float magnitude = sqrtf((float)(gx * gx + gy * gy));
    
    // Нормализуем значение в диапазон [0, 255]
    magnitude = fminf(magnitude, 255.0f);
    
    output[y * width + x] = (unsigned char)magnitude;
}

void applySobelGPU(unsigned char* h_input, unsigned char* h_output, 
                   int width, int height) {
    // Выделяем память на GPU
    unsigned char* d_input;
    unsigned char* d_output;
    
    size_t imageSize = width * height * sizeof(unsigned char);
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, imageSize));
    CUDA_CHECK(cudaMalloc((void**)&d_output, imageSize));
    
    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    
    // Настраиваем размеры блоков и сетки
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Запускаем kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Проверяем ошибки kernel
    CUDA_CHECK(cudaGetLastError());
    
    // Вычисляем время выполнения
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU processing time: %.3f ms\n", milliseconds);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Копируем результат обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

