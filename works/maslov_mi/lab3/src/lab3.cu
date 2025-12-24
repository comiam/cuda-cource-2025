#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>

#define CUDA_CHECK(err)                                                                    \
    do {                                                                                   \
        cudaError_t _e = (err);                                                            \
        if (_e != cudaSuccess) {                                                           \
            std::printf("%s in %s at line %d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNEL_NUM 3
#define TILE_SIZE 32
#define SHARED_WIDTH (TILE_SIZE + 2)

using byte = unsigned char;

__constant__ int SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

void convert_gray(const float* P_in, float* P_out, int N, int M) {
    int total = N * M;
    for (int i = 0; i < total; i++) {
        int idx = i * 3;
        P_out[i] = 0.21f * P_in[idx] + 0.71f * P_in[idx + 1] + 0.07f * P_in[idx + 2];
    }
}

__global__ void sobel_tiled_kernel(const byte* orig, byte* out, int width, int height) {
    extern __shared__ unsigned char sdata[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    int s_idx = (ty + 1) * SHARED_WIDTH + (tx + 1);
    if (x < width && y < height) {
        sdata[s_idx] = orig[y * width + x];
    } else {
        sdata[s_idx] = 0;
    }

    if (tx == 0) {
        int gx = x - 1;
        unsigned char v = (gx >= 0 && y < height) ? orig[y * width + gx] : 0;
        sdata[(ty + 1) * SHARED_WIDTH] = v;
    }
    if (tx == blockDim.x - 1) {
        int gx = x + 1;
        unsigned char v = (gx < width && y < height) ? orig[y * width + gx] : 0;
        sdata[(ty + 1) * SHARED_WIDTH + (tx + 2)] = v;
    }
    if (ty == 0) {
        int gy = y - 1;
        unsigned char v = (gy >= 0 && x < width) ? orig[gy * width + x] : 0;
        sdata[(tx + 1)] = v;
    }
    if (ty == blockDim.y - 1) {
        int gy = y + 1;
        unsigned char v = (gy < height && x < width) ? orig[gy * width + x] : 0;
        sdata[(ty + 2) * SHARED_WIDTH + (tx + 1)] = v;
    }

    if (tx == 0 && ty == 0) {
        int gx = x - 1, gy = y - 1;
        unsigned char v = (gx >= 0 && gy >= 0) ? orig[gy * width + gx] : 0;
        sdata[0] = v;
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        int gx = x + 1, gy = y - 1;
        unsigned char v = (gx < width && gy >= 0) ? orig[gy * width + gx] : 0;
        sdata[tx + 2] = v;
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        int gx = x - 1, gy = y + 1;
        unsigned char v = (gx >= 0 && gy < height) ? orig[gy * width + gx] : 0;
        sdata[(ty + 2) * SHARED_WIDTH] = v;
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int gx = x + 1, gy = y + 1;
        unsigned char v = (gx < width && gy < height) ? orig[gy * width + gx] : 0;
        sdata[(ty + 2) * SHARED_WIDTH + (tx + 2)] = v;
    }

    __syncthreads();

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int sx_c = tx + 1;
        int sy_c = ty + 1;

        float dx = 0.0f;
        float dy = 0.0f;

        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int sidx = (sy_c + ky) * SHARED_WIDTH + (sx_c + kx);
                float val = static_cast<float>(sdata[sidx]);
                dx += SOBEL_X[ky + 1][kx + 1] * val;
                dy += SOBEL_Y[ky + 1][kx + 1] * val;
            }
        }

        float mag = sqrtf(dx * dx + dy * dy);
        out[y * width + x] = static_cast<unsigned char>(fminf(mag, 255.0f));
    } else if (x < width && y < height) {
        out[y * width + x] = 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <input_image_path>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    int width = 0, height = 0, channels_in_file = 0;
    unsigned char* loaded = stbi_load(input_path, &width, &height, &channels_in_file, CHANNEL_NUM);
    if (!loaded) {
        std::fprintf(stderr, "Failed to load image '%s'\n", input_path);
        return 1;
    }

    size_t pixels = static_cast<size_t>(width) * height;

    std::vector<float> in(3 * pixels);
    std::vector<float> gray(pixels);
    for (size_t i = 0; i < pixels * 3; ++i) in[i] = static_cast<float>(loaded[i]);
    stbi_image_free(loaded);

    convert_gray(in.data(), gray.data(), height, width);

    std::vector<byte> gray_u8(pixels);
    for (size_t i = 0; i < pixels; ++i) {
        float v = std::max(0.0f, std::min(255.0f, gray[i]));
        gray_u8[i] = static_cast<byte>(v);
    }

    byte *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, pixels));
    CUDA_CHECK(cudaMalloc(&d_out, pixels));
    CUDA_CHECK(cudaMemcpy(d_in, gray_u8.data(), pixels, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    size_t shared_bytes = SHARED_WIDTH * SHARED_WIDTH * sizeof(unsigned char);

    sobel_tiled_kernel<<<grid, block, shared_bytes>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<byte> out(pixels);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, pixels, cudaMemcpyDeviceToHost));

    const char* out_name = "image_sobel.png";
    stbi_write_png(out_name, width, height, 1, out.data(), width);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
