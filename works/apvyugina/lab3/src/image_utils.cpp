#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "image_utils.h"
#include <iostream>
#include <vector>



Image loadImage(const std::string& filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    
    if (!data) {
        throw std::runtime_error("Не удалось загрузить изображение: " + filename);
    }
    
    Image img(width, height, channels);
    img.data.assign(data, data + width * height * channels);
    
    stbi_image_free(data);
    return img;
}

void saveImage(const Image& img, const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    
    int success = 0;
    if (ext == "png" || ext == "PNG") {
        success = stbi_write_png(filename.c_str(), img.width, img.height, 
                                 img.channels, img.data.data(), img.width * img.channels);
    } else if (ext == "bmp" || ext == "BMP") {
        success = stbi_write_bmp(filename.c_str(), img.width, img.height, 
                                 img.channels, img.data.data());
    } else {
        throw std::runtime_error("Неподдерживаемый формат: " + ext);
    }
    
    if (!success) {
        throw std::runtime_error("Не удалось сохранить изображение: " + filename);
    }
}

Image rgbToGrayscale(const Image& rgb) {
    if (rgb.channels != 3) {
        throw std::runtime_error("Изображение должно быть RGB");
    }
    
    Image gray(rgb.width, rgb.height, 1);
    
    for (int i = 0; i < rgb.width * rgb.height; ++i) {
        // Формула для конвертации RGB в grayscale
        gray.data[i] = static_cast<uint8_t>(
            0.299f * rgb.data[i * 3] + 
            0.587f * rgb.data[i * 3 + 1] + 
            0.114f * rgb.data[i * 3 + 2]
        );
    }
    
    return gray;
}


std::vector<float> normalize(const std::vector<unsigned char>& data) {
    std::vector<float> result(data.size());

    for(size_t i = 0; i < data.size(); ++i) {
        result[i] = static_cast<float>(data[i]) / 255.0f;
    }

    return result;
}

std::vector<unsigned char> denormalize(const std::vector<float>& data) {
    std::vector<unsigned char> result(data.size());

    for(size_t i = 0; i < data.size(); ++i) {
        result[i] = static_cast<unsigned char>(data[i] * 255.0f + 0.5f); // Округление
    }

    return result;
}