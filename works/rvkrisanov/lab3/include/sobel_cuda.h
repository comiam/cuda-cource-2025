#pragma once
#include <cuda_runtime.h>

void launch_sobel_filter(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    cudaStream_t stream = 0);
