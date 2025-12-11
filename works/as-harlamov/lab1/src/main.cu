#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void compute_circle(char* grid, int width, int height, int cx, int cy, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int dx = x - cx;
        int dy = y - cy;
        grid[y * width + x] = (dx * dx + dy * dy <= radius * radius);
    }
}

int main() {
    const int width = 80;
    const int height = 24;
    const int cx = width / 2;
    const int cy = height / 2;
    const int radius = 10;

    std::vector<char> h_grid(width * height);
    char* d_grid;
    cudaMalloc(&d_grid, width * height * sizeof(char));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    compute_circle<<<grid, block>>>(d_grid, width, height, cx, cy, radius);
    cudaDeviceSynchronize();

    cudaMemcpy(h_grid.data(), d_grid, width * height * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(d_grid);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << (h_grid[y * width + x] ? '*' : ' ');
        }
        std::cout << '\n';
    }

    return 0;
}
