//
// PGM image I/O functions
//

#ifndef PGM_IO_H
#define PGM_IO_H

// Структура для хранения изображения
struct Image {
    int width;
    int height;
    unsigned char* data;
};

// Загрузить PGM изображение из файла
Image loadPGM(const char* filename);

// Сохранить PGM изображение в файл
void savePGM(const char* filename, const Image& img);

// Освободить память изображения
void freeImage(Image& img);

#endif // PGM_IO_H

