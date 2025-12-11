//
// Universal image I/O implementation (PGM, PNG, BMP)
//

#include "../headers/image_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Простая реализация загрузки BMP
static Image loadBMP(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // Читаем заголовок BMP
    unsigned char header[54];
    if (fread(header, 1, 54, file) != 54) {
        fprintf(stderr, "Error: Invalid BMP file %s\n", filename);
        fclose(file);
        exit(1);
    }

    // Проверяем сигнатуру BMP
    if (header[0] != 'B' || header[1] != 'M') {
        fprintf(stderr, "Error: %s is not a BMP file\n", filename);
        fclose(file);
        exit(1);
    }

    Image img;
    // Читаем размеры из заголовка (little-endian)
    img.width = *(int*)&header[18];
    img.height = *(int*)&header[22];
    int bitsPerPixel = *(short*)&header[28];
    int dataOffset = *(int*)&header[10];

    // Поддерживаем только 24-bit и 32-bit BMP
    if (bitsPerPixel != 24 && bitsPerPixel != 32) {
        fprintf(stderr, "Error: Unsupported BMP format (only 24-bit and 32-bit supported)\n");
        fclose(file);
        exit(1);
    }

    int channels = bitsPerPixel / 8;
    int rowSize = ((img.width * channels + 3) / 4) * 4; // Выравнивание по 4 байта
    int imageSize = rowSize * img.height;

    // Выделяем память для RGB данных
    unsigned char* rgbData = (unsigned char*)malloc(imageSize);
    if (!rgbData) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        exit(1);
    }

    // Переходим к началу данных изображения
    fseek(file, dataOffset, SEEK_SET);
    if (fread(rgbData, 1, imageSize, file) != (size_t)imageSize) {
        fprintf(stderr, "Error: Failed to read BMP image data\n");
        free(rgbData);
        fclose(file);
        exit(1);
    }

    fclose(file);

    // Конвертируем RGB в grayscale
    img.data = (unsigned char*)malloc(img.width * img.height);
    if (!img.data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(rgbData);
        exit(1);
    }

    // BMP хранится в перевернутом виде (снизу вверх)
    for (int y = 0; y < img.height; y++) {
        int srcY = img.height - 1 - y; // Переворачиваем
        for (int x = 0; x < img.width; x++) {
            int srcIdx = (srcY * rowSize) + (x * channels);
            int dstIdx = y * img.width + x;
            
            // Конвертация RGB в grayscale (BGR порядок в BMP)
            unsigned char b = rgbData[srcIdx];
            unsigned char g = rgbData[srcIdx + 1];
            unsigned char r = rgbData[srcIdx + 2];
            
            // Формула для grayscale: 0.299*R + 0.587*G + 0.114*B
            img.data[dstIdx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }

    free(rgbData);
    printf("Loaded BMP image: %dx%d\n", img.width, img.height);
    return img;
}

// Простая реализация сохранения BMP
static void saveBMP(const char* filename, const Image& img) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        exit(1);
    }

    int rowSize = ((img.width * 3 + 3) / 4) * 4; // Выравнивание по 4 байта
    int imageSize = rowSize * img.height;
    int fileSize = 54 + imageSize; // 54 байта заголовок

    // Заголовок BMP
    unsigned char header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    *(int*)&header[2] = fileSize;
    *(int*)&header[10] = 54; // Смещение данных
    *(int*)&header[14] = 40; // Размер заголовка
    *(int*)&header[18] = img.width;
    *(int*)&header[22] = img.height;
    *(short*)&header[26] = 1; // Плоскости
    *(short*)&header[28] = 24; // Бит на пиксель
    *(int*)&header[34] = imageSize;

    fwrite(header, 1, 54, file);

    // Записываем данные (BMP хранится снизу вверх, BGR)
    unsigned char* row = (unsigned char*)malloc(rowSize);
    for (int y = img.height - 1; y >= 0; y--) {
        for (int x = 0; x < img.width; x++) {
            unsigned char gray = img.data[y * img.width + x];
            row[x * 3] = gray;     // B
            row[x * 3 + 1] = gray; // G
            row[x * 3 + 2] = gray; // R
        }
        // Заполняем padding нулями
        for (int x = img.width * 3; x < rowSize; x++) {
            row[x] = 0;
        }
        fwrite(row, 1, rowSize, file);
    }
    free(row);

    fclose(file);
    printf("Saved BMP image: %s (%dx%d)\n", filename, img.width, img.height);
}

// Загрузка PGM
static Image loadPGM(const char* filename) {
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

    fgetc(file); // Пропускаем пробельный символ

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

// Сохранение PGM
static void savePGM(const char* filename, const Image& img) {
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

// Загрузка PNG через stb_image (если доступна)
static Image loadPNG(const char* filename) {
    #ifdef STB_IMAGE_AVAILABLE
    #define STB_IMAGE_IMPLEMENTATION
    #include "../headers/stb_image.h"
    
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    
    if (!data) {
        fprintf(stderr, "Error: Cannot load PNG image %s: %s\n", filename, stbi_failure_reason());
        exit(1);
    }

    Image img;
    img.width = width;
    img.height = height;
    img.data = (unsigned char*)malloc(width * height);

    if (!img.data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        stbi_image_free(data);
        exit(1);
    }

    // Конвертируем в grayscale
    if (channels == 1) {
        memcpy(img.data, data, width * height);
    } else {
        rgbToGrayscale(data, img.data, width, height, channels);
    }

    stbi_image_free(data);
    printf("Loaded PNG image: %dx%d (channels: %d)\n", width, height, channels);
    return img;
    #else
    fprintf(stderr, "Error: PNG support requires stb_image library.\n");
    fprintf(stderr, "Please run: powershell -ExecutionPolicy Bypass -File download_stb.ps1\n");
    fprintf(stderr, "Or download stb_image.h from https://github.com/nothings/stb\n");
    fprintf(stderr, "and place it in the headers/ directory, then recompile with -DSTB_IMAGE_AVAILABLE\n");
    exit(1);
    #endif
}

// Сохранение PNG через stb_image_write (если доступна)
static void savePNG(const char* filename, const Image& img) {
    #ifdef STB_IMAGE_AVAILABLE
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "../headers/stb_image_write.h"
    
    // Сохраняем как grayscale PNG
    if (!stbi_write_png(filename, img.width, img.height, 1, img.data, img.width)) {
        fprintf(stderr, "Error: Cannot save PNG image %s\n", filename);
        exit(1);
    }
    
    printf("Saved PNG image: %s (%dx%d)\n", filename, img.width, img.height);
    #else
    fprintf(stderr, "Error: PNG save support requires stb_image_write library.\n");
    fprintf(stderr, "Please run: powershell -ExecutionPolicy Bypass -File download_stb.ps1\n");
    fprintf(stderr, "Or download stb_image_write.h from https://github.com/nothings/stb\n");
    fprintf(stderr, "and place it in the headers/ directory, then recompile with -DSTB_IMAGE_AVAILABLE\n");
    exit(1);
    #endif
}

ImageFormat getImageFormat(const char* filename) {
    const char* ext = strrchr(filename, '.');
    if (!ext) {
        return FORMAT_UNKNOWN;
    }

    if (strcmp(ext, ".pgm") == 0 || strcmp(ext, ".PGM") == 0) {
        return FORMAT_PGM;
    } else if (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0) {
        return FORMAT_PNG;
    } else if (strcmp(ext, ".bmp") == 0 || strcmp(ext, ".BMP") == 0) {
        return FORMAT_BMP;
    }
    
    return FORMAT_UNKNOWN;
}

Image loadImage(const char* filename) {
    ImageFormat format = getImageFormat(filename);
    
    switch (format) {
        case FORMAT_PGM:
            return loadPGM(filename);
        case FORMAT_PNG:
            return loadPNG(filename);
        case FORMAT_BMP:
            return loadBMP(filename);
        default:
            fprintf(stderr, "Error: Unsupported image format. Supported: .pgm, .png, .bmp\n");
            exit(1);
    }
}

void saveImage(const char* filename, const Image& img) {
    ImageFormat format = getImageFormat(filename);
    
    switch (format) {
        case FORMAT_PGM:
            savePGM(filename, img);
            break;
        case FORMAT_PNG:
            savePNG(filename, img);
            break;
        case FORMAT_BMP:
            saveBMP(filename, img);
            break;
        default:
            fprintf(stderr, "Error: Unsupported output format. Supported: .pgm, .png, .bmp\n");
            exit(1);
    }
}

void freeImage(Image& img) {
    if (img.data) {
        free(img.data);
        img.data = nullptr;
    }
    img.width = 0;
    img.height = 0;
}

void rgbToGrayscale(unsigned char* rgb, unsigned char* gray, int width, int height, int channels) {
    for (int i = 0; i < width * height; i++) {
        unsigned char r = rgb[i * channels];
        unsigned char g = (channels > 1) ? rgb[i * channels + 1] : r;
        unsigned char b = (channels > 2) ? rgb[i * channels + 2] : r;
        
        // Формула для grayscale: 0.299*R + 0.587*G + 0.114*B
        gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
