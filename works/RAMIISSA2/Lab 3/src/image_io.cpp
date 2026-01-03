#include "image_io.h"
#include <fstream>
#include <iostream>
#include <sstream>

// a function to skip the comments
static void skip_comments(std::istream& is) {
    while (is.peek() == '#') {
        is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

// a function to load a pgm image
bool load_pgm(const std::string& filename, Image& img) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open PGM file: " << filename << std::endl;
        return false;
    }

    std::string magic;
    file >> magic;
    if (magic != "P5") {
        std::cerr << "Unsupported PGM format (must be P5)" << std::endl;
        return false;
    }

    skip_comments(file);
    file >> img.width >> img.height;

    skip_comments(file);
    int maxval;
    file >> maxval;
    if (maxval != 255) {
        std::cerr << "Unsupported max value: " << maxval << std::endl;
        return false;
    }

    file.get(); // consume single whitespace after header

    img.data.resize(img.width * img.height);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());

    if (!file) {
        std::cerr << "Error reading PGM pixel data" << std::endl;
        return false;
    }

    return true;
}

// a function to save a pgm image
bool save_pgm(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return false;
    }

    file << "P5\n";
    file << img.width << " " << img.height << "\n";
    file << "255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());

    return true;
}