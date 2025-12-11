#include <cuda_runtime.h>


// CUDA kernel: determines if a thread's pixel belongs to the triangle
__global__ void drawTriangle(char* output, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x >= width || y >= height) return;


    int t_height = 2 * height / 3;
    int t_width = 2 * t_height - 1;
    if (t_width >= width)
    {
        t_width = 2 * width / 3;
        t_height = (t_width + 1) / 2;
    }
    // Triangle boundaries
    int left = (width - t_width) / 2;
    int right = left + t_width - 1;

    int top = (height - t_height) / 2;
    int bottom = top + t_height - 1;

    bool isCanvesBorder =
        (x == 0 && y >= 0 && y <= height - 1) ||
        (x == width - 1 && y >= 0 && y <= height - 1) ||
        (y == 0 && x >= 0 && x <= width - 1) ||
        (y == height - 1 && x >= 0 && x <= width - 1);

    bool isBorder =
        (x - y == right - bottom && y >= top && y <= bottom) ||
        (x + y == left + bottom && y >= top && y <= bottom) ||
        (y == bottom && x >= left && x <= right);

    output[y * width + x] = isCanvesBorder ? '.' : isBorder ? '*' : ' ';
}