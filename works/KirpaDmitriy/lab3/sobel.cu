#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TILE_SIZE 16 
#define BLOCK_SIZE (TILE_SIZE + 2) 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Image {
    int width;
    int height;
    std::vector<unsigned char> data;
};

Image loadImage(const std::string& filename) {
    int w, h, channels;
    unsigned char* raw_data = stbi_load(filename.c_str(), &w, &h, &channels, 1);
    
    if (raw_data == nullptr) {
        throw std::runtime_error("Failed to load image: " + filename + ". Reason: " + stbi_failure_reason());
    }

    Image img;
    img.width = w;
    img.height = h;
    img.data.assign(raw_data, raw_data + w * h);
    
    stbi_image_free(raw_data);
    
    return img;
}

void savePNG(const std::string& filename, const Image& img) {
    int stride = img.width * 1;
    int result = stbi_write_png(filename.c_str(), img.width, img.height, 1, img.data.data(), stride);
    
    if (result == 0) {
        throw std::runtime_error("Failed to write PNG file: " + filename);
    }
}

__global__ void SobelShared(const unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char smem[TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x_out = blockIdx.x * TILE_SIZE + tx;
    int y_out = blockIdx.y * TILE_SIZE + ty;

    int x_topLeft = blockIdx.x * TILE_SIZE - 1;
    int y_topLeft = blockIdx.y * TILE_SIZE - 1;

    int tile_dim_shared = TILE_SIZE + 2;
    int num_pixels_to_load = tile_dim_shared * tile_dim_shared;
    int num_threads = TILE_SIZE * TILE_SIZE;
    int thread_id = ty * TILE_SIZE + tx;

    #unroll
    for (int i = thread_id; i < num_pixels_to_load; i += num_threads) {
        int smem_y = i / tile_dim_shared;
        int smem_x = i % tile_dim_shared;
        int global_y = y_topLeft + smem_y;
        int global_x = x_topLeft + smem_x;

        smem[smem_y][smem_x] = (global_y >= 0 && global_y < height && global_x >= 0 && global_x < width)?
            input[global_y * width + global_x] : 0;
    }

    __syncthreads();

    if (x_out < width && y_out < height) {
        if (x_out > 0 && x_out < width - 1 && y_out > 0 && y_out < height - 1) {
            int s_r = ty + 1;
            int s_c = tx + 1;

            float Gx = 
                (-1) * smem[s_r - 1][s_c - 1] + (1) * smem[s_r - 1][s_c + 1] +
                (-2) * smem[s_r][s_c - 1]     + (2) * smem[s_r][s_c + 1] +
                (-1) * smem[s_r + 1][s_c - 1] + (1) * smem[s_r + 1][s_c + 1];

            float Gy = 
                (-1) * smem[s_r - 1][s_c - 1] + (-2) * smem[s_r - 1][s_c] + (-1) * smem[s_r - 1][s_c + 1] +
                (1)  * smem[s_r + 1][s_c - 1] + (2)  * smem[s_r + 1][s_c] + (1)  * smem[s_r + 1][s_c + 1];

            float val = sqrtf(Gx * Gx + Gy * Gy);
            output[y_out * width + x_out] = (val > 255.0f) ? 255 : static_cast<unsigned char>(val);
        } else {
            output[y_out * width + x_out] = 0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.png> <output.png>" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    Image inputImg;
    
    try {
        std::cout << "Loading " << inputFile << "..." << std::endl;
        inputImg = loadImage(inputFile);
        std::cout << "Image loaded: " << inputImg.width << "x" << inputImg.height << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    size_t imgSize = inputImg.data.size() * sizeof(unsigned char);
    
    unsigned char *d_in, *d_out;
    gpuErrchk(cudaMalloc(&d_in, imgSize));
    gpuErrchk(cudaMalloc(&d_out, imgSize));
    gpuErrchk(cudaMemcpy(d_in, inputImg.data.data(), imgSize, cudaMemcpyHostToDevice));

    Image outputImg;
    outputImg.width = inputImg.width;
    outputImg.height = inputImg.height;
    outputImg.data.resize(inputImg.width * inputImg.height);
    
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((inputImg.width + dimBlock.x - 1) / dimBlock.x, 
                 (inputImg.height + dimBlock.y - 1) / dimBlock.y);

    std::cout << "Processing on GPU..." << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    SobelShared<<<dimGrid, dimBlock>>>(d_in, d_out, inputImg.width, inputImg.height);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU Time: " << ms << " ms" << std::endl;
    
    gpuErrchk(cudaMemcpy(outputImg.data.data(), d_out, imgSize, cudaMemcpyDeviceToHost));
    
    try {
        savePNG(outputFile, outputImg);
        std::cout << "Result saved to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving: " << e.what() << std::endl;
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
