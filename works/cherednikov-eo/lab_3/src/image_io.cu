#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../include/image_io.cuh"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"


unsigned char* load_pgm(const char* filename, int* width, int* height) {
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


void save_pgm(const char* filename, unsigned char* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return;
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, file);
    fclose(file);
}


unsigned char* rgb_to_grayscale(unsigned char* rgb, int width, int height, int channels) {
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


unsigned char* load_image(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* data = stbi_load(filename, width, height, &channels, 0);

    if (!data) {
        fprintf(stderr, "Error: Cannot load image %s\n", filename);
        return NULL;
    }

    printf("Loaded image: %s (%dx%d, %d channels)\n", filename, *width, *height, channels);

    unsigned char* gray = rgb_to_grayscale(data, *width, *height, channels);
    stbi_image_free(data);

    return gray;
}


void save_png(const char* filename, unsigned char* data, int width, int height) {
    stbi_write_png(filename, width, height, 1, data, width);
}

void save_image(const char* filename, unsigned char* data, int width, int height) {
    const char* ext = strrchr(filename, '.');
    if (!ext) {
        fprintf(stderr, "Error: Cannot determine file format from filename %s\n", filename);
        return;
    }
    ext++; // Skip the dot
    
    if (strcmp(ext, "pgm") == 0 || strcmp(ext, "PGM") == 0) {
        save_pgm(filename, data, width, height);
    } else if (strcmp(ext, "png") == 0 || strcmp(ext, "PNG") == 0) {
        save_png(filename, data, width, height);
    } else {
        fprintf(stderr, "Error: Unsupported output format: %s\n", ext);
    }
}

