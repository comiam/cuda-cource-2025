#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

#include "../include/circle.h"
#include "../include/good_circle.h"
#include "../include/square.h"
#include "../include/triangle.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;
const int REPEATS = 1000;

float benchmarkSquare(dim3 grid, dim3 block, char* devBuffer);
float benchmarkCircle(dim3 grid, dim3 block, char* devBuffer);
float benchmarkGoodCircle(dim3 grid, dim3 block, char* devBuffer);
float benchmarkTriangle(dim3 grid, dim3 block, char* devBuffer);

// --------------------------------------------
// Utility to time circle kernel
// --------------------------------------------
float benchmarkSquare(dim3 grid, dim3 block, char* devBuffer) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < REPEATS; i++) {
        drawSquare << <grid, block >> > (devBuffer, WIDTH, HEIGHT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    return ms / REPEATS;
}

// --------------------------------------------
// Utility to time circle kernel
// --------------------------------------------
float benchmarkCircle(dim3 grid, dim3 block, char* devBuffer) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < REPEATS; i++) {
        drawCircle << <grid, block >> > (devBuffer, WIDTH, HEIGHT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    return ms / REPEATS;
}

// --------------------------------------------
// Utility to time circle kernel
// --------------------------------------------
float benchmarkGoodCircle(dim3 grid, dim3 block, char* devBuffer) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < REPEATS; i++) {
        drawGoodCircle << <grid, block >> > (devBuffer, WIDTH, HEIGHT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    return ms / REPEATS;
}

// --------------------------------------------
// Utility to time triangle kernel
// --------------------------------------------
float benchmarkTriangle(dim3 grid, dim3 block, char* devBuffer) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < REPEATS; i++) {
        drawTriangle << <grid, block >> > (devBuffer, WIDTH, HEIGHT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    return ms / REPEATS;
}

int main() {
    size_t bufferBytes = WIDTH * HEIGHT * sizeof(float);
    char* devBuffer;
    cudaMalloc(&devBuffer, bufferBytes);

    std::vector<dim3> blockSizes = {
        dim3(8, 8),
        dim3(16, 16),
        dim3(32, 32)
    };

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\nBlock Size \t| Square (ms) \t| Circle (ms) \t| Good Circle (ms) \t| Triangle (ms)\n";
    std::cout << "---------------------------------------------------------------------------------------\n";

    for (auto block : blockSizes) {
        dim3 grid(
            (WIDTH + block.x - 1) / block.x,
            (HEIGHT + block.y - 1) / block.y
        );

        float tSquare = benchmarkSquare(grid, block, devBuffer);
        float tCircle = benchmarkCircle(grid, block, devBuffer);
        float tGoodCircle = benchmarkGoodCircle(grid, block, devBuffer);
        float tTriangle = benchmarkTriangle(grid, block, devBuffer);

        std::cout << block.x << "x" << block.y
            << "\t\t| "
            << tSquare << "\t\t| "
            << tCircle << "\t\t| "
            << tGoodCircle << "\t\t\t| "
            << tTriangle << "\n";
    }

    cudaFree(devBuffer);

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}
