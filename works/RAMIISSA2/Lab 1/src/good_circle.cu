#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>


// CUDA kernel for drawing a circle outline
__global__ void drawGoodCircle(char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Circle parameters
    int cx = width / 2;
    int cy = height / 2;
    int radius = 5 * min(width, height) / 12;

    int dx = (x - cx) / 2;
    int dy = y - cy;

    int dist2 = dx * dx + dy * dy;
    int r2 = radius * radius;

    // Outline thickness
    int thickness = 2 * radius / 3;

    bool isCanvesBorder =
        (x == 0 && y >= 0 && y <= height - 1) ||
        (x == width - 1 && y >= 0 && y <= height - 1) ||
        (y == 0 && x >= 0 && x <= width - 1) ||
        (y == height - 1 && x >= 0 && x <= width - 1);

    bool isBorder = abs(dist2 - r2) < thickness;

    output[y * width + x] = isCanvesBorder ? '.' : isBorder ? '*' : ' ';
}