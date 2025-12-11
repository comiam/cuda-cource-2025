//
// Sobel operator header file
//

#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <cuda_runtime.h>
#include "error_check.cuh"

// CUDA kernel для применения оператора Собеля
__global__ void sobelKernel(unsigned char* input, unsigned char* output, 
                            int width, int height);

// Функция для применения оператора Собеля на GPU
void applySobelGPU(unsigned char* h_input, unsigned char* h_output, 
                   int width, int height);

#endif // SOBEL_CUH

