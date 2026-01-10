#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void draw(char *buf, int W, int H, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    float cx = W / 2.0f;
    float cy = H / 2.0f;

    float dx = x - cx;
    float dy = (y - cy) * 2.0f;  // компенсация пропорций

    float d = sqrtf(dx*dx + dy*dy);

    buf[y*W + x] = (fabsf(d - radius) < 1.9f) ? '*' : ' ';
}

int main(int argc, char **argv)
{
    int size = 20;  // значение по умолчанию

    if (argc > 1) {
        size = atoi(argv[1]);
        if (size < 10) size = 10;
        if (size > 100) size = 100;
    }

    const int W = size * 2;     // ширина в символах
    const int H = size;         // высота (меньше в 2 раза из-за пропорций)
    const float radius = size * 0.9f;   // ← вот здесь радиус зависит от size!

    printf("Размер: %d × %d, радиус ≈ %.1f\n\n", W, H, radius);

    char *buf, *dbuf;
    buf = (char*)malloc(W*H);
    cudaMalloc(&dbuf, W*H);
    cudaMemset(dbuf, ' ', W*H);

    dim3 block(16, 8);
    dim3 grid((W + 15)/16, (H + 7)/8);

    draw<<<grid, block>>>(dbuf, W, H, radius);
    cudaDeviceSynchronize();

    cudaMemcpy(buf, dbuf, W*H, cudaMemcpyDeviceToHost);

    for(int y = 0; y < H; y++) {
        for(int x = 0; x < W; x++)
            putchar(buf[y*W + x]);
        putchar('\n');
    }

    cudaFree(dbuf);
    free(buf);
    return 0;
}
