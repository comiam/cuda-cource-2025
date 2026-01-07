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

__constant__ float sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant__ float sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output,
                                           int width, int height) {
    __shared__ unsigned char shared_tile[TILE_SIZE+2][TILE_SIZE+2];

    for (int dy = 0; dy < TILE_SIZE+2; dy += TILE_SIZE) {
        for (int dx = 0; dx < TILE_SIZE+2; dx += TILE_SIZE) {
            int ty_shared = threadIdx.y + dy;
            int tx_shared = threadIdx.x + dx;
            
            if (ty_shared < TILE_SIZE+2 && tx_shared < TILE_SIZE+2) {
                int col_shared = blockIdx.x * TILE_SIZE + tx_shared - 1;
                int row_shared = blockIdx.y * TILE_SIZE + ty_shared - 1;
                
                if (col_shared >= 0 && col_shared < width && row_shared >= 0 && row_shared < height) {
                    shared_tile[ty_shared][tx_shared] = input[row_shared * width + col_shared];
                } else {
                    shared_tile[ty_shared][tx_shared] = 0;
                }
            }
        }
    }
    
    __syncthreads();

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    float gx = 0, gy = 0;
    
    if (tx > 0 && tx < TILE_SIZE + 1 && ty > 0 && ty < TILE_SIZE + 1 && 
        col < width && row < height && 
        col > 0 && col < width - 1 && row > 0 && row < height - 1) {
        
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                float pixel = shared_tile[ty + 1 + i][tx + 1 + j];
                gx += pixel * sobel_x[i + 1][j + 1];
                gy += pixel * sobel_y[i + 1][j + 1];
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        if (magnitude > 255.0f) magnitude = 255.0f;
        if (magnitude < 0.0f) magnitude = 0.0f;
        
        output[row * width + col] = static_cast<unsigned char>(magnitude);
    }
    else if (col >= 0 && col < width && row >= 0 && row < height) {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int c = col + j;
                int r = row + i;
                
                if (c >= 0 && c < width && r >= 0 && r < height) {
                    float pixel = input[r * width + c];
                    gx += pixel * sobel_x[i + 1][j + 1];
                    gy += pixel * sobel_y[i + 1][j + 1];
                }
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);
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
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + TILE_SIZE - 1) / 16, (height + TILE_SIZE - 1) / 16);
    
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