#include <cuda_runtime.h>


// CUDA kernel: determines if a thread's pixel belongs to the square
__global__ void drawSquare(char* output, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x >= width || y >= height) return;

    int s_height = 2 * height / 3;
    int s_width = 2 * s_height - 1;
    if (s_width >= width)
    {
        s_width = 2 * width / 3;
        s_height = (s_width + 1) / 2;
    }
    // Square boundaries
    int left = (width - s_width) / 2;
    int right = left + s_width -1;

    int top = (height - s_height) / 2;
    int bottom = top + s_height - 1;

    bool isCanvesBorder =
        (x == 0 && y >= 0 && y <= height - 1) ||
        (x == width - 1 && y >= 0 && y <= height - 1) ||
        (y == 0 && x >= 0 && x <= width - 1) ||
        (y == height - 1 && x >= 0 && x <= width - 1);

    bool isBorder =
        (x == left && y >= top && y <= bottom) ||
        (x == right && y >= top && y <= bottom) ||
        (y == top && x >= left && x <= right) ||
        (y == bottom && x >= left && x <= right);

    output[y * width + x] = isCanvesBorder ? '.' : isBorder ? '*' : ' ';
}