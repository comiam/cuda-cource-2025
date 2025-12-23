#include <cstdio>
#include <cuda_runtime.h>
#include "../include/error_check.cuh"

__global__ void draw_circle(
    char *canvas,
    const int canvas_width,
    const int canvas_height,
    const float radius,
    const float thickness,
    const float center_x,
    const float center_y)
{
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_x >= canvas_width || pixel_y >= canvas_height)
        return;

    int idx = pixel_y * canvas_width + pixel_x;

    float distance = sqrtf(
        (pixel_x - center_x) * (pixel_x - center_x) +
        (pixel_y - center_y) * (pixel_y - center_y));

    if (fabsf(distance - radius) <= thickness / 2.0f)
    {
        canvas[idx] = '*';
    }
    else
    {
        canvas[idx] = '.';
    }
}

int main()
{
    const int CANVAS_WIDTH = 64;
    const int CANVAS_HEIGHT = 64;
    const float RADIUS = 10.0f;
    const float THICKNESS = 5.0f;
    const float CENTER_X = CANVAS_WIDTH / 2.0f;
    const float CENTER_Y = CANVAS_HEIGHT / 2.0f;
    const size_t CANVAS_SIZE = CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(char);

    char *canvas_host = (char *)malloc(CANVAS_SIZE);

    char *canvas_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&canvas_gpu, CANVAS_SIZE));

    dim3 block(16, 16, 1);
    dim3 grid((CANVAS_WIDTH + block.x - 1) / block.x,
              (CANVAS_HEIGHT + block.y - 1) / block.y);

    draw_circle<<<grid, block>>>(canvas_gpu, CANVAS_WIDTH, CANVAS_HEIGHT, 
                                  RADIUS, THICKNESS, CENTER_X, CENTER_Y);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(canvas_host, canvas_gpu, CANVAS_SIZE, cudaMemcpyDeviceToHost));

    for (int y = 0; y < CANVAS_HEIGHT; ++y)
    {
        for (int x = 0; x < CANVAS_WIDTH; ++x)
        {
            printf("%c", canvas_host[y * CANVAS_WIDTH + x]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(canvas_gpu));
    free(canvas_host);

    return 0;
}
