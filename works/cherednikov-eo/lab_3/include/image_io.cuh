//
// Image I/O functions for PGM and PNG formats
//

#ifndef CUDA_COURCE_2025_IMAGE_IO_CUH
#define CUDA_COURCE_2025_IMAGE_IO_CUH

#pragma once


unsigned char* load_image(const char* filename, int* width, int* height);
void save_image(const char* filename, unsigned char* data, int width, int height);


unsigned char* load_pgm(const char* filename, int* width, int* height);
void save_pgm(const char* filename, unsigned char* data, int width, int height);

void save_png(const char* filename, unsigned char* data, int width, int height);

#endif //CUDA_COURCE_2025_IMAGE_IO_CUH
