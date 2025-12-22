#include "../include/sobel_cuda.h"
#include "../include/error_check.cuh"
#include <cmath>

#define BLOCK_SIZE 32
#define OUTPUT_TILE_SIZE 30 

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output, int width, int height)
{
    __shared__ unsigned char shared_memory_tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int output_x = blockIdx.x * OUTPUT_TILE_SIZE + thread_x - 1; 
    int output_y = blockIdx.y * OUTPUT_TILE_SIZE + thread_y - 1;

    bool is_inside_image = (output_x >= 0 && output_x < width && output_y >= 0 && output_y < height);
    
    shared_memory_tile[thread_y][thread_x] = is_inside_image ? input[output_y * width + output_x] : 0;

    __syncthreads();

    if (thread_x > 0 && thread_x < BLOCK_SIZE - 1 && thread_y > 0 && thread_y < BLOCK_SIZE - 1) 
    {
        if (is_inside_image) 
        {
            unsigned char nw = shared_memory_tile[thread_y - 1][thread_x - 1];
            unsigned char n  = shared_memory_tile[thread_y - 1][thread_x];
            unsigned char ne = shared_memory_tile[thread_y - 1][thread_x + 1];
            unsigned char w  = shared_memory_tile[thread_y][thread_x - 1];
            unsigned char e  = shared_memory_tile[thread_y][thread_x + 1];
            unsigned char sw = shared_memory_tile[thread_y + 1][thread_x - 1];
            unsigned char s  = shared_memory_tile[thread_y + 1][thread_x];
            unsigned char se = shared_memory_tile[thread_y + 1][thread_x + 1];

            int Gx = (-1 * nw + 0 * n + 1 * ne) + 
                     (-2 * w  + 0     + 2 * e ) + 
                     (-1 * sw + 0 * s + 1 * se);

            int Gy = (-1 * nw - 2 * n - 1 * ne) + 
                     ( 0 * w  + 0     + 0 * e ) + 
                     ( 1 * sw + 2 * s + 1 * se);

            float magnitude = sqrtf(static_cast<float>(Gx * Gx + Gy * Gy));
            
            magnitude = (magnitude > 255.0f) ? 255.0f : magnitude;

            output[output_y * width + output_x] = static_cast<unsigned char>(magnitude);
        }
    }
}

void launch_sobel_filter(
    const unsigned char* device_input,
    unsigned char* device_output,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 grid((width + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, 
              (height + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE);

    sobel_kernel<<<grid, block, 0, stream>>>(device_input, device_output, width, height);
    CUDA_CHECK(cudaGetLastError());
}
