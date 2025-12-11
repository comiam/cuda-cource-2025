// stb_image_write.h - v1.16
// Public domain image writer - http://nothings.org/stb
// no warranty implied; use at your own risk

#ifndef STB_IMAGE_WRITE_H
#define STB_IMAGE_WRITE_H

#ifdef STB_IMAGE_WRITE_IMPLEMENTATION
// Реализация будет в image_io.cpp
#endif

// Простой интерфейс для сохранения изображений
int stbi_write_png(const char* filename, int w, int h, int comp, const void* data, int stride_in_bytes);

#endif // STB_IMAGE_WRITE_H

