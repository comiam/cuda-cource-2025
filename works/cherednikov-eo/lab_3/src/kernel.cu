#include <cuda_runtime.h>
#include <cmath>


__global__ void sobel_kernel(
    const unsigned char* input,
    unsigned char* output,
    unsigned width,
    unsigned height
) {
    __shared__ float shared_data[18][18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x + tx;
    int by = blockIdx.y * blockDim.y + ty;

    // Загрузка центральных пикселей блока в shared memory
    if (bx < width && by < height) {
        shared_data[ty + 1][tx + 1] = static_cast<float>(input[by * width + bx]);
    } else {
        shared_data[ty + 1][tx + 1] = 0.0f;
    }

    // Загрузка верхней границы блока
    if (ty == 0) {
        int y_up = by - 1;
        if (y_up >= 0 && bx < width) {
            shared_data[0][tx + 1] = static_cast<float>(input[y_up * width + bx]);
        } else {
            shared_data[0][tx + 1] = 0.0f;
        }
    }
    
    // Загрузка нижней границы блока
    if (ty == blockDim.y - 1) {
        int y_down = by + 1;
        if (y_down < height && bx < width) {
            shared_data[ty + 2][tx + 1] = static_cast<float>(input[y_down * width + bx]);
        } else {
            shared_data[ty + 2][tx + 1] = 0.0f;
        }
    }

    // Загрузка левой границы блока
    if (tx == 0) {
        int x_left = bx - 1;
        if (x_left >= 0 && by < height) {
            shared_data[ty + 1][0] = static_cast<float>(input[by * width + x_left]);
        } else {
            shared_data[ty + 1][0] = 0.0f;
        }
    }
    
    // Загрузка правой границы блока
    if (tx == blockDim.x - 1) {
        int x_right = bx + 1;
        if (x_right < width && by < height) {
            shared_data[ty + 1][tx + 2] = static_cast<float>(input[by * width + x_right]);
        } else {
            shared_data[ty + 1][tx + 2] = 0.0f;
        }
    }

    // Загрузка угловых пикселей
    if (tx == 0 && ty == 0) {
        int x = bx - 1, y = by - 1;
        shared_data[0][0] = (x >= 0 && y >= 0) ? static_cast<float>(input[y * width + x]) : 0.0f;
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        int x = bx + 1, y = by - 1;
        shared_data[0][tx + 2] = (x < width && y >= 0) ? static_cast<float>(input[y * width + x]) : 0.0f;
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        int x = bx - 1, y = by + 1;
        shared_data[ty + 2][0] = (x >= 0 && y < height) ? static_cast<float>(input[y * width + x]) : 0.0f;
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int x = bx + 1, y = by + 1;
        shared_data[ty + 2][tx + 2] = (x < width && y < height) ? static_cast<float>(input[y * width + x]) : 0.0f;
    }
    __syncthreads();

    // Применение оператора Собеля только к внутренним пикселям
    // (не к граничным, так как для них нет полного окружения 3x3)
    if (bx >= 1 && bx < width - 1 && by >= 1 && by < height - 1) {
        // Чтение окна 3x3 из shared memory
        float p00 = shared_data[ty][tx];
        float p01 = shared_data[ty][tx + 1];
        float p02 = shared_data[ty][tx + 2];
        float p10 = shared_data[ty + 1][tx];
        float p12 = shared_data[ty + 1][tx + 2];
        float p20 = shared_data[ty + 2][tx];
        float p21 = shared_data[ty + 2][tx + 1];
        float p22 = shared_data[ty + 2][tx + 2];

        // Вычисление градиента по оси X
        // Маска Gx: [-1  0  1]
        //           [-2  0  2]
        //           [-1  0  1]
        float Gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;

        // Вычисление градиента по оси Y
        // Маска Gy: [-1 -2 -1]
        //           [ 0  0  0]
        //           [ 1  2  1]
        float Gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;

        float mag = sqrtf(Gx * Gx + Gy * Gy);
        mag = fminf(fmaxf(mag, 0.0f), 255.0f);

        output[by * width + bx] = static_cast<unsigned char>(mag);
    } else {
        // Граничные пиксели устанавливаются в 0
        if (bx < width && by < height) {
            output[by * width + bx] = 0;
        }
    }
}