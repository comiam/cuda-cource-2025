#ifndef IMAGE_UTILS
#define IMAGE_UTILS

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

struct Image {
    std::vector<uint8_t> data;
    int width;
    int height;
    int channels;
    
    Image(int w, int h, int ch) : width(w), height(h), channels(ch) {
        data.resize(w * h * ch);
    }
};

Image loadImage(const std::string& filename);
void saveImage(const Image& img, const std::string& filename);
Image rgbToGrayscale(const Image& rgb);

std::vector<float> normalize(const std::vector<unsigned char>& data);
std::vector<unsigned char> denormalize(const std::vector<float>& data);

#endif