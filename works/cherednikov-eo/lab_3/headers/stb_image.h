// stb_image.h - v2.27
// Public domain image loader - http://nothings.org/stb
// no warranty implied; use at your own risk

#ifndef STB_IMAGE_H
#define STB_IMAGE_H

#ifdef STB_IMAGE_IMPLEMENTATION
// Реализация будет в image_io.cpp
#endif

// Простой интерфейс для загрузки изображений
unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_file, int desired_channels);
void stbi_image_free(void* retval_from_stbi_load);
const char* stbi_failure_reason(void);

#endif // STB_IMAGE_H

