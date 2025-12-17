#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include "../include/image_io.cuh"


#if defined(__has_include) && __has_include("../include/stb_image.h") && __has_include("../include/stb_image_write.h")
    #define STB_IMAGE_IMPLEMENTATION
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "../include/stb_image.h"
    #include "../include/stb_image_write.h"
    #define PNG_SUPPORT_AVAILABLE
#endif

int load_pgm(const char* filename, unsigned char** data, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return -1;
    }

    char magic[3];
    if (fscanf(file, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: file must be in PGM format (P5)\n");
        fclose(file);
        return -1;
    }

    int c = fgetc(file);
    while (c == '#') {
        while (fgetc(file) != '\n');
        c = fgetc(file);
    }
    ungetc(c, file);

    if (fscanf(file, "%d %d", width, height) != 2) {
        fprintf(stderr, "Error: failed to read image dimensions\n");
        fclose(file);
        return -1;
    }

    int max_val;
    if (fscanf(file, "%d", &max_val) != 1 || max_val != 255) {
        fprintf(stderr, "Error: maximum value must be 255\n");
        fclose(file);
        return -1;
    }

    fgetc(file);

    *data = (unsigned char*)malloc(*width * *height * sizeof(unsigned char));
    if (!*data) {
        fprintf(stderr, "Error: failed to allocate memory\n");
        fclose(file);
        return -1;
    }

    if (fread(*data, sizeof(unsigned char), *width * *height, file) != *width * *height) {
        fprintf(stderr, "Error: failed to read image data\n");
        free(*data);
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

int save_pgm(const char* filename, unsigned char* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: failed to create file %s\n", filename);
        return -1;
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);
    if (fwrite(data, sizeof(unsigned char), width * height, file) != width * height) {
        fprintf(stderr, "Error: failed to write image data\n");
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}


static const char* get_file_extension(const char* filename) {
    const char* dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}


static int ends_with(const char* str, const char* suffix) {
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (str_len < suffix_len) return 0;
    for (size_t i = 0; i < suffix_len; i++) {
        if (tolower(str[str_len - suffix_len + i]) != tolower(suffix[i])) {
            return 0;
        }
    }
    return 1;
}

int load_png(const char* filename, unsigned char** data, int* width, int* height) {
#ifdef PNG_SUPPORT_AVAILABLE
    int channels;
    unsigned char* img_data = stbi_load(filename, width, height, &channels, 1);
    if (!img_data) {
        fprintf(stderr, "Error: failed to load PNG image %s: %s\n", filename, stbi_failure_reason());
        return -1;
    }


    *data = (unsigned char*)malloc(*width * *height * sizeof(unsigned char));
    if (!*data) {
        fprintf(stderr, "Error: failed to allocate memory\n");
        stbi_image_free(img_data);
        return -1;
    }

    if (channels == 1) {
        memcpy(*data, img_data, *width * *height);
    } else {

        for (int i = 0; i < *width * *height; i++) {
            // Use standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            unsigned char r = img_data[i * channels];
            unsigned char g = (channels > 1) ? img_data[i * channels + 1] : 0;
            unsigned char b = (channels > 2) ? img_data[i * channels + 2] : 0;
            (*data)[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }

    stbi_image_free(img_data);
    return 0;
#else
    fprintf(stderr, "Error: PNG support not available. Please include stb_image.h\n");
    return -1;
#endif
}

int save_png(const char* filename, unsigned char* data, int width, int height) {
#ifdef PNG_SUPPORT_AVAILABLE
    int result = stbi_write_png(filename, width, height, 1, data, width);
    if (!result) {
        fprintf(stderr, "Error: failed to save PNG image %s\n", filename);
        return -1;
    }
    return 0;
#else
    fprintf(stderr, "Error: PNG support not available. Please include stb_image_write.h\n");
    return -1;
#endif
}

int load_image(const char* filename, unsigned char** data, int* width, int* height) {
    const char* ext = get_file_extension(filename);
    
    if (strcmp(ext, "pgm") == 0 || strcmp(ext, "PGM") == 0) {
        return load_pgm(filename, data, width, height);
    } else if (strcmp(ext, "png") == 0 || strcmp(ext, "PNG") == 0) {
        return load_png(filename, data, width, height);
    } else {
        FILE* file = fopen(filename, "rb");
        if (!file) {
            fprintf(stderr, "Error: failed to open file %s\n", filename);
            return -1;
        }
        
        char magic[8] = {0};
        fread(magic, 1, 8, file);
        fclose(file);

        if (magic[0] == 0x89 && magic[1] == 'P' && magic[2] == 'N' && magic[3] == 'G') {
            return load_png(filename, data, width, height);
        }
        else if (magic[0] == 'P' && magic[1] == '5') {
            return load_pgm(filename, data, width, height);
        }
        else {
            fprintf(stderr, "Error: unsupported image format. Supported formats: PGM, PNG\n");
            return -1;
        }
    }
}

int save_image(const char* filename, unsigned char* data, int width, int height) {
    const char* ext = get_file_extension(filename);
    
    if (strcmp(ext, "pgm") == 0 || strcmp(ext, "PGM") == 0) {
        return save_pgm(filename, data, width, height);
    } else if (strcmp(ext, "png") == 0 || strcmp(ext, "PNG") == 0) {
        return save_png(filename, data, width, height);
    } else {
        fprintf(stderr, "Error: unsupported output format. Supported formats: PGM, PNG\n");
        return -1;
    }
}

