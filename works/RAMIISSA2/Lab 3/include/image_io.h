#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct Image {
    int width;
    int height;
    std::vector<uint8_t> data;
};

// Load PGM (P5) image
bool load_pgm(const std::string& filename, Image& img);

// Save PGM (P5) image
bool save_pgm(const std::string& filename, const Image& img);
