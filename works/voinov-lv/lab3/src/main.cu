#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <cstdio>

#define CUDA_CHECK(call)                                                         \
do {                                                                             \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
        fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__,                \
                cudaGetErrorString(err__));                                      \
        std::abort();                                                            \
    }                                                                            \
} while(0)

#define TILE_SIZE 16

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output,
                                           int width, int height) {
    __shared__ unsigned char shared_tile[TILE_SIZE+2][TILE_SIZE+2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    if (col < width && row < height) {
        shared_tile[ty][tx] = input[row * width + col];
    } else {
        shared_tile[ty][tx] = 0;
    }
    
    __syncthreads();
    
    if (tx > 0 && tx < TILE_SIZE + 1 && ty > 0 && ty < TILE_SIZE + 1 && 
        col < width && row < height && 
        col > 0 && col < width - 1 && row > 0 && row < height - 1) {
        
        float sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        float sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        
        float gx = 0, gy = 0;
        
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                float pixel = shared_tile[ty + i][tx + j];
                gx += pixel * sobel_x[i + 1][j + 1];
                gy += pixel * sobel_y[i + 1][j + 1];
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        if (magnitude > 255.0f) magnitude = 255.0f;
        if (magnitude < 0.0f) magnitude = 0.0f;
        
        output[row * width + col] = static_cast<unsigned char>(magnitude);
    }
}

std::vector<unsigned char> rgb_to_gray(unsigned char* data, int width, int height, int channels) {
    std::vector<unsigned char> gray_image(width * height);
    
    if (channels == 1) {
        for (int i = 0; i < width * height; i++) {
            gray_image[i] = data[i];
        }
    } else if (channels >= 3) {
        for (int i = 0; i < width * height; i++) {
            unsigned char r = data[i * channels];
            unsigned char g = data[i * channels + 1];
            unsigned char b = data[i * channels + 2];
            gray_image[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
    
    return gray_image;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    int width, height, channels;
    unsigned char* data = stbi_load(input_file.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Error: Failed to load image: " << input_file << std::endl;
        return 1;
    }
    
    std::vector<unsigned char> gray_image = rgb_to_gray(data, width, height, channels);
    stbi_image_free(data);
    
    unsigned char *d_input, *d_output;
    size_t image_size = width * height * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    
    CUDA_CHECK(cudaMemcpy(d_input, gray_image.data(), image_size, cudaMemcpyHostToDevice));
    
    dim3 block_size(18, 18);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    
    sobel_kernel<<<grid_size, block_size>>>(d_input, d_output, width, height);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<unsigned char> output_image(width * height);
    CUDA_CHECK(cudaMemcpy(output_image.data(), d_output, image_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    output_file += ".png";
    bool write_success = stbi_write_png(output_file.c_str(), width, height, 1, output_image.data(), width);
    
    if (!write_success) {
        std::cerr << "Error: Failed to save image: " << output_file << std::endl;
        return 1;
    }
    
    std::cout << "Saved: " << output_file << std::endl;
    
    return 0;
}