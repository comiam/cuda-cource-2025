#include "../headers/utils.h"
#include "../headers/error-check.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void sobel_kernel(
    const unsigned char* input,
    unsigned char* output,
    unsigned width,
    unsigned height
);

int main(int argc, char* argv[]) {
    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::vector<unsigned char> h_input;
    unsigned width, height;

    if (!loadImage(input_path, h_input, width, height)) {
        return 1;
    }

    size_t img_size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), img_size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    sobel_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Время выполнения Sobel: " << ms << " мс (размер: " << width << "x" << height << ")" << std::endl;

    std::vector<unsigned char> h_output(width * height);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, img_size, cudaMemcpyDeviceToHost));

    if (!saveImage(output_path, h_output, width, height)) {
        return 1;
    }

    std::cout << "Результат сохранён в: " << output_path << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}