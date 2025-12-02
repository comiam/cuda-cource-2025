#include <iostream>
#include <cuda_runtime.h>

#include "../include/circle.h"
#include "../include/good_circle.h"
#include "../include/square.h"
#include "../include/triangle.h"

#define WIDTH 41
#define HEIGHT 21

int main() {
    char* d_output;
    char* h_output = new char[WIDTH * HEIGHT];
    int size = WIDTH * HEIGHT * sizeof(char);

    cudaMalloc(&d_output, size);

    int choice;
    std::cout << "Choose shape:\n";
    std::cout << "1. Square\n2. Circle\n3. Good Circle\n4. Triangle\n> ";
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (WIDTH + blockSize.x - 1) / blockSize.x,
        (HEIGHT + blockSize.y - 1) / blockSize.y
    );

    if (choice == 1) {
        drawSquare << <gridSize, blockSize >> > (d_output, WIDTH, HEIGHT);
    }
    else if (choice == 2) {
        drawCircle << <gridSize, blockSize >> > (d_output, WIDTH, HEIGHT);
    }
    else if (choice == 3) {
        drawGoodCircle << <gridSize, blockSize >> > (d_output, WIDTH, HEIGHT);
    }
    else if (choice == 4) {
        drawTriangle << <gridSize, blockSize >> > (d_output, WIDTH, HEIGHT);
    }
    else {
        std::cout << "Invalid choice.\n";
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << h_output[y * WIDTH + x];
        }
        std::cout << "\n";
    }

    cudaFree(d_output);
    delete[] h_output;

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}
