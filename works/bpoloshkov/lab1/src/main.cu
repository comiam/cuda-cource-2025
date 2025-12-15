#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void drawCircle(char* canvas, int width, int height, float radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float cx = width / 2.0f;
    float cy = height / 2.0f;

    float dx = x - cx;
    float dy = (y - cy) * 2.0f;
    float dist = sqrtf(dx * dx + dy * dy);

    char pixel = (fabsf(dist - radius) <= 1.5f) ? '*' : ' ';
    canvas[y * width + x] = pixel;
}

int main(int argc, char** argv) {
    int size = 20;

    if (argc > 1) {
        size = atoi(argv[1]);
        if (size < 10 || size > 100) {
            fprintf(stderr, "Error: size must be between 10 and 100\n");
            return 1;
        }
    }

    int width = size;
    int height = size / 2;
    float radius = height - 2.0f;
    size_t bufSize = width * height * sizeof(char);

    char* d_canvas;
    CUDA_CHECK(cudaMalloc(&d_canvas, bufSize));
    CUDA_CHECK(cudaMemset(d_canvas, ' ', bufSize));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    drawCircle<<<gridSize, blockSize>>>(d_canvas, width, height, radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    char* h_canvas = (char*)malloc(bufSize);
    CUDA_CHECK(cudaMemcpy(h_canvas, d_canvas, bufSize, cudaMemcpyDeviceToHost));

    printf("Circle, size: %dx%d\n\n", width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            putchar(h_canvas[y * width + x]);
        }
        putchar('\n');
    }

    free(h_canvas);
    cudaFree(d_canvas);

    return 0;
}
