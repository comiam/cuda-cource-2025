#include <cuda_runtime.h>
#include <cmath>

__global__ void sobel_kernel(
    const unsigned char* input,
    unsigned char* output,
    unsigned width,
    unsigned height
) {
    // Размер shared-памяти: (1+16+1) x (1+16+1) = 18x18
    __shared__ unsigned char shared_data[18][18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x + tx;
    int by = blockIdx.y * blockDim.y + ty;

    if (bx < width && by < height) {
        shared_data[ty + 1][tx + 1] = input[by * width + bx];
    } else {
        shared_data[ty + 1][tx + 1] = 0;
    }

    if (ty == 0) {
        int y = by - 1;
        if (y >= 0 && bx < width) {
            shared_data[0][tx + 1] = input[y * width + bx];
        } else {
            shared_data[0][tx + 1] = 0;
        }
    }
    if (ty == blockDim.y - 1) {
        int y = by + 1;
        if (y < height && bx < width) {
            shared_data[ty + 2][tx + 1] = input[y * width + bx];
        } else {
            shared_data[ty + 2][tx + 1] = 0;
        }
    }

    if (tx == 0) {
        int x = bx - 1;
        if (x >= 0 && by < height) {
            shared_data[ty + 1][0] = input[by * width + x];
        } else {
            shared_data[ty + 1][0] = 0;
        }
    }
    if (tx == blockDim.x - 1) {
        int x = bx + 1;
        if (x < width && by < height) {
            shared_data[ty + 1][tx + 2] = input[by * width + x];
        } else {
            shared_data[ty + 1][tx + 2] = 0;
        }
    }

    if (tx == 0 && ty == 0) {
        int x = bx - 1, y = by - 1;
        shared_data[0][0] = (x >= 0 && y >= 0) ? input[y * width + x] : 0;
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        int x = bx + 1, y = by - 1;
        shared_data[0][tx + 2] = (x < width && y >= 0) ? input[y * width + x] : 0;
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        int x = bx - 1, y = by + 1;
        shared_data[ty + 2][0] = (x >= 0 && y < height) ? input[y * width + x] : 0;
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int x = bx + 1, y = by + 1;
        shared_data[ty + 2][tx + 2] = (x < width && y < height) ? input[y * width + x] : 0;
    }

    __syncthreads();

    if (bx >= 1 && bx < width - 1 && by >= 1 && by < height - 1) {
        float p00 = static_cast<float>(shared_data[ty][tx]);
        float p01 = static_cast<float>(shared_data[ty][tx + 1]);
        float p02 = static_cast<float>(shared_data[ty][tx + 2]);
        float p10 = static_cast<float>(shared_data[ty + 1][tx]);
        float p11 = static_cast<float>(shared_data[ty + 1][tx + 1]);
        float p12 = static_cast<float>(shared_data[ty + 1][tx + 2]);
        float p20 = static_cast<float>(shared_data[ty + 2][tx]);
        float p21 = static_cast<float>(shared_data[ty + 2][tx + 1]);
        float p22 = static_cast<float>(shared_data[ty + 2][tx + 2]);

        // Gx = [-1 0 1]
        //      [-2 0 2]
        //      [-1 0 1]
        float Gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;

        // Gy = [-1 -2 -1]
        //      [ 0  0  0]
        //      [ 1  2  1]
        float Gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;

        float mag = sqrtf(Gx * Gx + Gy * Gy);
        if (mag > 255.0f) mag = 255.0f;
        output[by * width + bx] = static_cast<unsigned char>(mag);
    } else if (bx < width && by < height) {
        // Пиксели по краю — чёрные (0)
        output[by * width + bx] = 0;
    }
}
