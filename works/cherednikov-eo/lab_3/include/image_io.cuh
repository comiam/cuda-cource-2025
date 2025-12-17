//
// Image I/O functions for PGM and PNG formats
//

#ifndef CUDA_COURCE_2025_IMAGE_IO_CUH
#define CUDA_COURCE_2025_IMAGE_IO_CUH

#pragma once

int load_image(const char* filename, unsigned char** data, int* width, int* height);
int save_image(const char* filename, unsigned char* data, int width, int height);


int load_pgm(const char* filename, unsigned char** data, int* width, int* height);
int save_pgm(const char* filename, unsigned char* data, int width, int height);


int load_png(const char* filename, unsigned char** data, int* width, int* height);
int save_png(const char* filename, unsigned char* data, int width, int height);

#endif //CUDA_COURCE_2025_IMAGE_IO_CUH

