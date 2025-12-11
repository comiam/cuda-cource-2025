//
// PGM image I/O implementation
//

#include "../headers/pgm_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

Image loadPGM(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    char magic[3];
    if (fscanf(file, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: %s is not a PGM P5 file\n", filename);
        fclose(file);
        exit(1);
    }

    // Пропускаем комментарии
    int c = fgetc(file);
    while (c == '#') {
        while (fgetc(file) != '\n');
        c = fgetc(file);
    }
    ungetc(c, file);

    Image img;
    if (fscanf(file, "%d %d", &img.width, &img.height) != 2) {
        fprintf(stderr, "Error: Invalid PGM header in %s\n", filename);
        fclose(file);
        exit(1);
    }

    int maxVal;
    if (fscanf(file, "%d", &maxVal) != 1) {
        fprintf(stderr, "Error: Invalid PGM header in %s\n", filename);
        fclose(file);
        exit(1);
    }

    // Пропускаем один пробельный символ после maxVal
    fgetc(file);

    img.data = (unsigned char*)malloc(img.width * img.height);
    if (!img.data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        exit(1);
    }

    size_t read = fread(img.data, 1, img.width * img.height, file);
    if (read != (size_t)(img.width * img.height)) {
        fprintf(stderr, "Error: Failed to read image data from %s\n", filename);
        free(img.data);
        fclose(file);
        exit(1);
    }

    fclose(file);
    printf("Loaded PGM image: %dx%d\n", img.width, img.height);
    return img;
}

void savePGM(const char* filename, const Image& img) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n255\n", img.width, img.height);
    fwrite(img.data, 1, img.width * img.height, file);
    fclose(file);
    printf("Saved PGM image: %s (%dx%d)\n", filename, img.width, img.height);
}

void freeImage(Image& img) {
    if (img.data) {
        free(img.data);
        img.data = nullptr;
    }
    img.width = 0;
    img.height = 0;
}

