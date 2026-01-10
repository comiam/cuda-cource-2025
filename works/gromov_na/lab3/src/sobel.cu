#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (expr);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "cuda error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// загрузка pgm-изображения
bool load_pgm(const std::string &filename, std::vector<unsigned char> &pixels, int &width, int &height)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "could not open input file: " << filename << "\n";
        return false;
    }

    std::string line;
    std::getline(file, line);
    if (line != "P5")
    {
        std::cerr << "unsupported format (not P5)\n";
        return false;
    }

    std::getline(file, line);
    while (line[0] == '#')
        std::getline(file, line); // пропуск комментариев

    std::istringstream iss(line);
    iss >> width >> height;

    std::getline(file, line);
    int max_val;
    std::istringstream iss2(line);
    iss2 >> max_val;

    pixels.resize(width * height);
    file.read(reinterpret_cast<char *>(pixels.data()), width * height);

    file.close();
    return true;
}

// сохранение pgm-изображения
bool save_pgm(const std::string &filename, const std::vector<unsigned char> &pixels, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "could not create output file: " << filename << "\n";
        return false;
    }

    file << "P5\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(pixels.data()), width * height);

    file.close();
    return true;
}

__global__ void sobel_filter(const unsigned char *input, unsigned char *output, int width, int height)
{
    const int tile_size = 16;
    const int rx = 1; // радиус окрестности

    // shared memory для оптимизации доступа к данным
    __shared__ float shared_tile[tile_size + 2 * rx][tile_size + 2 * rx];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + rx;
    int ty = threadIdx.y + rx;

    // загрузка центральной части в shared memory - каждый поток загружает один элемент
    if (row < height && col < width)
    {
        shared_tile[ty][tx] = static_cast<float>(input[row * width + col]);
    }
    else
    {
        // для потоков за пределами изображения загружаем 0
        shared_tile[ty][tx] = 0.0f;
    }

    // загрузка граничных элементов (halo elements) - потоки на границе блока загружают дополнительные элементы
    // левые и правые границы
    if (threadIdx.x < rx)
    {
        int halo_col_left = blockIdx.x * blockDim.x - rx + threadIdx.x;
        int halo_col_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

        if (halo_col_left >= 0 && halo_col_left < width)
        {
            int src_row = min(max(row, 0), height - 1);
            int src_col = min(max(halo_col_left, 0), width - 1);
            shared_tile[ty][threadIdx.x] = static_cast<float>(input[src_row * width + src_col]);
        }
        else
        {
            shared_tile[ty][threadIdx.x] = 0.0f;
        }

        if (halo_col_right < width)
        {
            int src_row = min(max(row, 0), height - 1);
            int src_col = min(max(halo_col_right, 0), width - 1);
            shared_tile[ty][tile_size + threadIdx.x + rx] = static_cast<float>(input[src_row * width + src_col]);
        }
        else
        {
            shared_tile[ty][tile_size + threadIdx.x + rx] = 0.0f;
        }
    }

    // верхние и нижние границы
    if (threadIdx.y < rx)
    {
        int halo_row_top = blockIdx.y * blockDim.y - rx + threadIdx.y;
        int halo_row_bottom = (blockIdx.y + 1) * blockDim.y + threadIdx.y;

        if (halo_row_top >= 0 && halo_row_top < height)
        {
            int src_row = min(max(halo_row_top, 0), height - 1);
            int src_col = min(max(col, 0), width - 1);
            shared_tile[threadIdx.y][tx] = static_cast<float>(input[src_row * width + src_col]);
        }
        else
        {
            shared_tile[threadIdx.y][tx] = 0.0f;
        }

        if (halo_row_bottom < height)
        {
            int src_row = min(max(halo_row_bottom, 0), height - 1);
            int src_col = min(max(col, 0), width - 1);
            shared_tile[tile_size + threadIdx.y + rx][tx] = static_cast<float>(input[src_row * width + src_col]);
        }
        else
        {
            shared_tile[tile_size + threadIdx.y + rx][tx] = 0.0f;
        }
    }

    __syncthreads();

    if (row < height && col < width)
    {
        // вычисление градиента по оси x (Gx)
        float gx = (-1.0f * shared_tile[ty - 1][tx - 1] + 1.0f * shared_tile[ty - 1][tx + 1]) + (-2.0f * shared_tile[ty][tx - 1] + 2.0f * shared_tile[ty][tx + 1]) + (-1.0f * shared_tile[ty + 1][tx - 1] + 1.0f * shared_tile[ty + 1][tx + 1]);

        // вычисление градиента по оси y (Gy)
        float gy = (-1.0f * shared_tile[ty - 1][tx - 1] - 2.0f * shared_tile[ty - 1][tx] + (-1.0f * shared_tile[ty - 1][tx + 1])) + (1.0f * shared_tile[ty + 1][tx - 1] + 2.0f * shared_tile[ty + 1][tx] + (1.0f * shared_tile[ty + 1][tx + 1]));

        // вычисление величины градиента
        float magnitude = sqrt(gx * gx + gy * gy);
        unsigned char result = static_cast<unsigned char>(min(magnitude, 255.0f));
        output[row * width + col] = result;
    }
}

int main(int argc, char *argv[])
{
    std::string input_file = "../data/bu.pgm";
    std::string output_file = "../data/edges.pgm";

    // обработка аргументов командной строки
    if (argc >= 2)
        input_file = argv[1];
    if (argc >= 3)
        output_file = argv[2];

    std::vector<unsigned char> pixels;
    int width, height;

    // загрузка входного изображения
    if (!load_pgm(input_file, pixels, width, height))
    {
        return 1;
    }

    std::cout << "loaded image: " << width << "x" << height << "\n";

    unsigned char *d_input, *d_output;
    size_t img_size = width * height * sizeof(unsigned char);

    // выделение памяти на gpu
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));

    // копирование данных на gpu
    CUDA_CHECK(cudaMemcpy(d_input, pixels.data(), img_size, cudaMemcpyHostToDevice));

    // настройка конфигурации ядра
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // вызов ядра оператора собеля
    sobel_filter<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // копирование результата обратно на cpu
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_output, img_size, cudaMemcpyDeviceToHost));

    // сохранение результата
    if (!save_pgm(output_file, pixels, width, height))
    {
        return 1;
    }

    std::cout << "result saved to: " << output_file << "\n";

    // освобождение gpu памяти
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}