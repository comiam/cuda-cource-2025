#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#include "../include/sobel_cuda.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)


float process_image_on_gpu(const unsigned char* host_input, unsigned char* host_output, int width, int height)
{
    size_t img_size = width * height * sizeof(unsigned char);
    unsigned char *device_input, *device_output;
    cudaStream_t stream;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaMalloc(&device_input, img_size));
    CUDA_CHECK(cudaMalloc(&device_output, img_size));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpyAsync(device_input, host_input, img_size, cudaMemcpyHostToDevice, stream));


    CUDA_CHECK(cudaEventRecord(start, stream));
    
    launch_sobel_filter(device_input, device_output, width, height, stream);

    CUDA_CHECK(cudaEventRecord(stop, stream));

    CUDA_CHECK(cudaMemcpyAsync(host_output, device_output, img_size, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path> [output_image_path]" << std::endl;
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = (argc > 2) ? argv[2] : "output_sobel.png";

    int width, height, channels;
    unsigned char* host_input = stbi_load(input_path, &width, &height, &channels, 1);
    if (!host_input) {
        std::cerr << "Error loading image: " << input_path << std::endl;
        return 1;
    }
    
    std::cout << "Image loaded: " << width << "x" << height << " (Grayscale)" << std::endl;

    unsigned char* host_output = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    std::cout << "Processing on GPU..." << std::endl;
    float time_ms = process_image_on_gpu(host_input, host_output, width, height);
    
    std::cout << "Done! Kernel time: " << time_ms << " ms" << std::endl;

    if (stbi_write_png(output_path, width, height, 1, host_output, width)) {
        std::cout << "Result saved to: " << output_path << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
    }

    stbi_image_free(host_input);
    free(host_output);

    return 0;
}
