//
// Universal image I/O functions (PGM, PNG, BMP)
//

#ifndef IMAGE_IO_H
#define IMAGE_IO_H

// Структура для хранения изображения
struct Image {
    int width;
    int height;
    unsigned char* data;
};

// Определить формат изображения по расширению файла
enum ImageFormat {
    FORMAT_PGM,
    FORMAT_PNG,
    FORMAT_BMP,
    FORMAT_UNKNOWN
};

ImageFormat getImageFormat(const char* filename);

// Универсальная загрузка изображения (PGM, PNG, BMP)
Image loadImage(const char* filename);

// Универсальное сохранение изображения (PGM, PNG, BMP)
void saveImage(const char* filename, const Image& img);

// Освободить память изображения
void freeImage(Image& img);

// Конвертировать RGB в grayscale
void rgbToGrayscale(unsigned char* rgb, unsigned char* gray, int width, int height, int channels);

#endif // IMAGE_IO_H

