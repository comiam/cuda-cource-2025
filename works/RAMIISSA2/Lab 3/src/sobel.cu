#include "image_io.h"
#include <cuda_runtime.h>
#include <iostream>

__global__
void sobel_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    // Compute pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ignore borders
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return;
    }

    int idx = y * width + x;

    // Sobel Gx
    int gx =
        -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
        -2 * input[y * width + (x - 1)]   + 2 * input[y * width + (x + 1)]
        -input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    // Sobel Gy
    int gy =
        -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
        +input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    // Gradient magnitude
    float magnitude = sqrtf(float(gx * gx + gy * gy));

    // Clamp to [0, 255]
    magnitude = fminf(255.0f, magnitude);

    output[idx] = static_cast<uint8_t>(magnitude);
}

int main() {
    // CPU Load
    Image input_img;
    if (!load_pgm("assets/sample_7.pgm", input_img)) {
        return 1;
    }

    std::cout << "Loaded image: "
        << input_img.width << "x" << input_img.height << std::endl;

    size_t imageSize = input_img.width * input_img.height * sizeof(uint8_t);

    // Allocate GPU memory
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Copy input CPU to GPU
    cudaMemcpy(
        d_input,
        input_img.data.data(),
        imageSize,
        cudaMemcpyHostToDevice
    );

    std::cout << "Image copied to GPU memory." << std::endl;

    // CUDA grid
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (input_img.width + blockSize.x - 1) / blockSize.x,
        (input_img.height + blockSize.y - 1) / blockSize.y
    );

    // Sobel kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sobel_kernel << <gridSize, blockSize >> > (
        d_input,
        d_output,
        input_img.width,
        input_img.height
        );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    std::cout << "GPU kernel time (global memory): "
        << elapsed_ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result GPU to CPU
    Image output_img;
    output_img.width = input_img.width;
    output_img.height = input_img.height;
    output_img.data.resize(input_img.width * input_img.height);

    cudaMemcpy(
        output_img.data.data(),
        d_output,
        imageSize,
        cudaMemcpyDeviceToHost
    );

    // Save output
    save_pgm("edges.pgm", output_img);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
