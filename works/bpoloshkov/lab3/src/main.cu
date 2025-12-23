#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define BLOCK_SIZE 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
           - 2 * input[y * width + (x - 1)]     + 2 * input[y * width + (x + 1)]
           - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
           + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
    output[y * width + x] = (magnitude > 255) ? 255 : magnitude;
}

unsigned char* readPGM(const char* filename, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        return NULL;
    }

    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: not a PGM P5 file\n");
        fclose(fp);
        return NULL;
    }

    int c;
    while ((c = fgetc(fp)) == '#') {
        while (fgetc(fp) != '\n');
    }
    ungetc(c, fp);

    int maxval;
    if (fscanf(fp, "%d %d %d", width, height, &maxval) != 3) {
        fprintf(stderr, "Error: invalid PGM header\n");
        fclose(fp);
        return NULL;
    }
    fgetc(fp);

    size_t size = (*width) * (*height);
    unsigned char* data = (unsigned char*)malloc(size);
    if (fread(data, 1, size, fp) != size) {
        fprintf(stderr, "Error: failed to read image data\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return data;
}

void writePGM(const char* filename, unsigned char* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot create file %s\n", filename);
        return;
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, fp);
    fclose(fp);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input.pgm> <output.pgm>\n", argv[0]);
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    int width, height;
    unsigned char* h_input = readPGM(inputFile, &width, &height);
    if (!h_input) return 1;

    printf("Image: %dx%d\n", width, height);

    size_t size = width * height * sizeof(unsigned char);
    unsigned char* h_output = (unsigned char*)malloc(size);

    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeMs;
    cudaEventElapsedTime(&timeMs, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    printf("Processing time: %.2f ms\n", timeMs);

    writePGM(outputFile, h_output, width, height);
    printf("Result saved to: %s\n", outputFile);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

