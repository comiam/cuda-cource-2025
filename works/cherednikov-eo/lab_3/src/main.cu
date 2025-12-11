//
// Main program for Sobel edge detection
//

#include <cstdio>
#include <cstdlib>
#include "../headers/image_io.h"
#include "../headers/sobel.cuh"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        fprintf(stderr, "Supported formats: .pgm, .png, .bmp\n");
        fprintf(stderr, "Example: %s input.png edges.bmp\n", argv[0]);
        fprintf(stderr, "Example: %s input.pgm edges.pgm\n", argv[0]);
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    // Загружаем изображение (поддерживаются PGM, PNG, BMP)
    printf("Loading image: %s\n", inputFile);
    Image inputImg = loadImage(inputFile);
    
    // Выделяем память для выходного изображения
    Image outputImg;
    outputImg.width = inputImg.width;
    outputImg.height = inputImg.height;
    outputImg.data = (unsigned char*)malloc(inputImg.width * inputImg.height);
    
    if (!outputImg.data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        freeImage(inputImg);
        return 1;
    }

    // Применяем оператор Собеля на GPU
    printf("Applying Sobel operator on GPU...\n");
    applySobelGPU(inputImg.data, outputImg.data, inputImg.width, inputImg.height);

    // Сохраняем результат (формат определяется по расширению)
    printf("Saving result: %s\n", outputFile);
    saveImage(outputFile, outputImg);

    // Освобождаем память
    freeImage(inputImg);
    freeImage(outputImg);

    printf("Edge detection completed successfully!\n");
    return 0;
}

