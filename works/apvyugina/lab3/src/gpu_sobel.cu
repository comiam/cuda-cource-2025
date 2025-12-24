#include <stdio.h>
#include <vector>
#include <chrono>
#include "image_utils.h"
#include <cuda_runtime.h>

// Константные маски Собеля в постоянной памяти
__constant__ float gx_const[3][3] = {
    {-1.f, 0.f, 1.f},
    {-2.f, 0.f, 2.f},
    {-1.f, 0.f, 1.f}
};

__constant__ float gy_const[3][3] = {
    {-1.f, -2.f, -1.f},
    {0.f, 0.f, 0.f},
    {1.f, 2.f, 1.f}
};

__global__ void sobelFilterTexture(float* output, cudaTextureObject_t texObj, int width, int height) {
    // Текущие индексы блока и нити
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    // Суммы интенсивностей
    float intensityX = 0.0f;
    float intensityY = 0.0f;
    
    // Центральные нормализированные координаты пикселя
    float centerU = (x + 0.5f)/width;
    float centerV = (y + 0.5f)/height;

    // Итерируем окно 3x3 вокруг центрального пикселя
    for(int dy=-1;dy<=1;++dy){
        for(int dx=-1;dx<=1;++dx){
            float nu = (centerU + dx/(float)width);
            float nv = (centerV + dy/(float)height);
        
            // Извлекаем значение пикселя из текстуры
            float value = tex2D<float>(texObj, nu, nv);
            
            intensityX += gx_const[dy+1][dx+1]*value;
            intensityY += gy_const[dy+1][dx+1]*value;
        }
    }

    // Рассчитываем абсолютную величину градиента
    float gradientMagnitude = sqrtf(intensityX*intensityX + intensityY*intensityY);
    output[y * width + x] = gradientMagnitude / (4.0f * sqrtf(2.0f));
}


std::vector<float> applySobelCuda(std::vector<float> input_pixels, int width, int height){
    float* input_pixels_array = input_pixels.data();

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(float);
    // Copy data located at address input_pixels_array in host memory to device memory
    cudaMemcpy2DToArray(cuArray, 0, 0, input_pixels_array, spitch, width * sizeof(float),
                        height, cudaMemcpyHostToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Allocate result of transformation in device memory
    float *output_d;
    cudaMalloc(&output_d, width * height * sizeof(float));

    // Invoke kernel
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);

    auto start_time = std::chrono::high_resolution_clock::now();
    sobelFilterTexture<<<numBlocks, threadsperBlock>>>(output_d, texObj, width, height);
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
    double cuda_duration = elapsed.count();
    printf("Время работы на GPU без переноса данных: %f секунд.\n", cuda_duration);

    // Copy data from device back to host
    float* output_h = (float *)std::malloc(sizeof(float) * width * height);
    cudaMemcpy(output_h, output_d, width * height * sizeof(float),
                cudaMemcpyDeviceToHost);

    // Destroy texture object
    cudaDestroyTextureObject(texObj);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output_d);

    std::vector<float>output_pixels(output_h, output_h + width * height);
    return output_pixels;
}