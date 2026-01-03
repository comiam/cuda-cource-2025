#include "image_io.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__
void sobel_shared_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    __shared__ uint8_t tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    int sx = tx + 1;
    int sy = ty + 1;

    // Load center pixels
    tile[sy][sx] = (x < width && y < height) ? input[y * width + x] : 0;

    // Load halo pixels for boundary threads
    if (tx == 0)
        tile[sy][0] = (x > 0) ? input[y * width + (x - 1)] : 0;

    if (tx == BLOCK_SIZE - 1)
        tile[sy][BLOCK_SIZE + 1] = (x + 1 < width) ? input[y * width + (x + 1)] : 0;

    if (ty == 0)
        tile[0][sx] = (y > 0) ? input[(y - 1) * width + x] : 0;

    if (ty == BLOCK_SIZE - 1)
        tile[BLOCK_SIZE + 1][sx] = (y + 1 < height) ? input[(y + 1) * width + x] : 0;

    // Load halo pixels for corners threads
    if (tx == 0 && ty == 0)
        tile[0][0] = (x > 0 && y > 0) ? input[(y - 1) * width + (x - 1)] : 0;

    if (tx == BLOCK_SIZE - 1 && ty == 0)
        tile[0][BLOCK_SIZE + 1] = (x + 1 < width && y > 0) ? input[(y - 1) * width + (x + 1)] : 0;

    if (tx == 0 && ty == BLOCK_SIZE - 1)
        tile[BLOCK_SIZE + 1][0] = (x > 0 && y + 1 < height) ? input[(y + 1) * width + (x - 1)] : 0;

    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1)
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = (x + 1 < width && y + 1 < height) ? input[(y + 1) * width + (x + 1)] : 0;

    // Synchronize threads
    __syncthreads();

    // Ignore borders
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        return;

    // Sobel Gx
    int gx =
        - tile[sy - 1][sx - 1] + tile[sy - 1][sx + 1]
        - 2 * tile[sy][sx - 1] + 2 * tile[sy][sx + 1]
        - tile[sy + 1][sx - 1] + tile[sy + 1][sx + 1];

    // Sobel Gy
    int gy =
        -tile[sy - 1][sx - 1] - 2 * tile[sy - 1][sx] - tile[sy - 1][sx + 1]
        + tile[sy + 1][sx - 1] + 2 * tile[sy + 1][sx] + tile[sy + 1][sx + 1];

    float magnitude = sqrtf(float(gx * gx + gy * gy));
    magnitude = fminf(255.0f, magnitude);

    output[y * width + x] = static_cast<uint8_t>(magnitude);
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
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (input_img.width + blockSize.x - 1) / blockSize.x,
        (input_img.height + blockSize.y - 1) / blockSize.y
    );

    // Sobel kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sobel_shared_kernel << <gridSize, blockSize >> > (
        d_input,
        d_output,
        input_img.width,
        input_img.height
        );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    std::cout << "GPU kernel time (shared memory): "
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
    save_pgm("edges_shared.pgm", output_img);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
