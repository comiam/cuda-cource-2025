#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

__constant__ int d_Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

#define TILE_SIZE 16

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char tile[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    
    // Load tile with halo
    int tile_x = x - 1;
    int tile_y = y - 1;
    tile_x = max(0, min(tile_x, width - 1));
    tile_y = max(0, min(tile_y, height - 1));
    tile[ty][tx] = input[tile_y * width + tile_x];
    
    if (tx < 2 && tx + TILE_SIZE < TILE_SIZE + 2) {
        int halo_x = max(0, min(x + TILE_SIZE - 1, width - 1));
        tile[ty][tx + TILE_SIZE] = input[tile_y * width + halo_x];
    }
    if (ty < 2 && ty + TILE_SIZE < TILE_SIZE + 2) {
        int halo_y = max(0, min(y + TILE_SIZE - 1, height - 1));
        tile[ty + TILE_SIZE][tx] = input[halo_y * width + tile_x];
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }
    
    int gx = 0, gy = 0;
    
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int pixel = tile[ty + ky][tx + kx];
            int kernelIndex = ky * 3 + kx;
            gx += pixel * d_Gx[kernelIndex];
            gy += pixel * d_Gy[kernelIndex];
        }
    }
    
    int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
    magnitude = min(255, max(0, magnitude));
    
    output[y * width + x] = (unsigned char)magnitude;
}

unsigned char* loadPGM(const char* filename, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    if (fscanf(file, "%2s", magic) != 1) {
        fprintf(stderr, "Error: Cannot read PGM magic number\n");
        fclose(file);
        return NULL;
    }
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Not a valid PGM file (P5 format expected)\n");
        fclose(file);
        return NULL;
    }
    
    char c = getc(file);
    while (c == '#') {
        while (getc(file) != '\n');
        c = getc(file);
    }
    ungetc(c, file);
    
    int maxval;
    if (fscanf(file, "%d %d %d", width, height, &maxval) != 3) {
        fprintf(stderr, "Error: Cannot read PGM dimensions\n");
        fclose(file);
        return NULL;
    }
    fgetc(file); 
    
    int size = (*width) * (*height);
    unsigned char* data = (unsigned char*)malloc(size);
    size_t bytes_read = fread(data, 1, size, file);
    if (bytes_read != (size_t)size) {
        fprintf(stderr, "Warning: Expected %d bytes, read %zu bytes\n", size, bytes_read);
    }
    
    fclose(file);
    return data;
}

void savePGM(const char* filename, unsigned char* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return;
    }
    
    fprintf(file, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, file);
    fclose(file);
}

unsigned char* rgbToGrayscale(unsigned char* rgb, int width, int height, int channels) {
    int size = width * height;
    unsigned char* gray = (unsigned char*)malloc(size);
    
    for (int i = 0; i < size; i++) {
        if (channels >= 3) {
            int r = rgb[i * channels + 0];
            int g = rgb[i * channels + 1];
            int b = rgb[i * channels + 2];
            gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        } else {
            gray[i] = rgb[i * channels];
        }
    }
    
    return gray;
}

unsigned char* loadImage(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* data = stbi_load(filename, width, height, &channels, 0);
    
    if (!data) {
        fprintf(stderr, "Error: Cannot load image %s\n", filename);
        return NULL;
    }
    
    printf("Loaded image: %s (%dx%d, %d channels)\n", filename, *width, *height, channels);
    
    unsigned char* gray = rgbToGrayscale(data, *width, *height, channels);
    stbi_image_free(data);
    
    return gray;
}

void savePNG(const char* filename, unsigned char* data, int width, int height) {
    stbi_write_png(filename, width, height, 1, data, width);
}

int main(int argc, char** argv) {
    const char* imageFiles[] = {
        "data/image.png",
        "data/MSI.png"
    };
    int numImages = sizeof(imageFiles) / sizeof(imageFiles[0]);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < numImages; i++) {
        const char* inputFile = imageFiles[i];
        
        int width, height;
        unsigned char* h_input = NULL;
        
        const char* ext = strrchr(inputFile, '.');
        if (ext && strcmp(ext, ".pgm") == 0) {
            h_input = loadPGM(inputFile, &width, &height);
            printf("Loaded PGM image: %s (%dx%d)\n", inputFile, width, height);
        } else {
            h_input = loadImage(inputFile, &width, &height);
        }
        
        if (!h_input) {
            fprintf(stderr, "Failed to load image %s, skipping...\n", inputFile);
            continue;
        }
        
        unsigned char *d_input, *d_output;
        size_t imageSize = width * height * sizeof(unsigned char);
        
        cudaMalloc(&d_input, imageSize);
        cudaMalloc(&d_output, imageSize);
        cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
        
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE,
                      (height + TILE_SIZE - 1) / TILE_SIZE);
        
        cudaEventRecord(start);
        sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Время на GPU: %.2f мс\n", milliseconds);
        
        unsigned char* h_output = (unsigned char*)malloc(imageSize);
        cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
        
        char outputFilePGM[512];
        char outputFilePNG[512];
        
        const char* filename = strrchr(inputFile, '/');
        if (!filename) filename = strrchr(inputFile, '\\');
        if (!filename) filename = inputFile;
        else filename++;
        
        snprintf(outputFilePGM, sizeof(outputFilePGM), "%.*s_sobel.pgm",
                 (int)(ext ? ext - filename : strlen(filename)), filename);
        snprintf(outputFilePNG, sizeof(outputFilePNG), "%.*s_sobel.png",
                 (int)(ext ? ext - filename : strlen(filename)), filename);
        
        savePGM(outputFilePGM, h_output, width, height);
        printf("Saved PGM: %s\n", outputFilePGM);
        
        int pgm_width, pgm_height;
        unsigned char* pgm_data = loadPGM(outputFilePGM, &pgm_width, &pgm_height);
        if (pgm_data) {
            savePNG(outputFilePNG, pgm_data, pgm_width, pgm_height);
            printf("Saved PNG: %s\n", outputFilePNG);
            free(pgm_data);
        } else {
            fprintf(stderr, "Error: Could not load PGM file %s for PNG conversion\n", outputFilePGM);
        }
        
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
