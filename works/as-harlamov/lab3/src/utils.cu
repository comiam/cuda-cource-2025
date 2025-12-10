#include "utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <limits>
#include "lodepng.h"


bool endsWith(const std::string& str, const std::string& suffix) {
    if (str.length() >= suffix.length()) {
        return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin(),
            [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    }
    return false;
}

bool loadPGM(const std::string& filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return false;
    }

    std::string magic;
    file >> magic;
    if (magic != "P5") {
        std::cerr << "Файл не является бинарным PGM (ожидался P5, найдено: " << magic << ")" << std::endl;
        return false;
    }

    char c;
    while (file.get(c)) {
        if (c == '#') {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else if (std::isspace(c)) {
            continue;
        } else {
            file.putback(c);
            break;
        }
    }

    if (!(file >> width >> height)) {
        std::cerr << "Не удалось прочитать ширину и высоту из PGM-файла." << std::endl;
        return false;
    }

    int maxval;
    if (!(file >> maxval)) {
        std::cerr << "Не удалось прочитать maxval из PGM-файла." << std::endl;
        return false;
    }

    if (maxval <= 0 || maxval > 255) {
        std::cerr << "Неподдерживаемый maxval: " << maxval << " (ожидалось 1..255)" << std::endl;
        return false;
    }

    file.get(c);
    while (file.get(c) && std::isspace(static_cast<unsigned char>(c))) {
    }
    if (file) {
        file.putback(c);
    }

    size_t expected_size = static_cast<size_t>(width) * static_cast<size_t>(height);
    image.resize(expected_size);
    file.read(reinterpret_cast<char*>(image.data()), expected_size);

    if (!file) {
        std::cerr << "Ошибка при чтении пиксельных данных из PGM." << std::endl;
        return false;
    }

    return true;
}

bool savePGM(const std::string& filename, const std::vector<unsigned char>& image, unsigned width, unsigned height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), image.size());
    return true;
}

bool loadPNG(const std::string& filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height) {
    std::vector<unsigned char> rgba;
    unsigned error = lodepng::decode(rgba, width, height, filename);
    if (error) {
        std::cerr << "Ошибка lodepng при загрузке " << filename << ": "
                  << lodepng_error_text(error) << std::endl;
        return false;
    }

    // RGBA -> Grayscale (0.299*R + 0.587*G + 0.114*B)
    image.resize(width * height);
    for (size_t i = 0; i < rgba.size(); i += 4) {
        float gray = 0.299f * rgba[i] + 0.587f * rgba[i + 1] + 0.114f * rgba[i + 2];
        image[i / 4] = static_cast<unsigned char>(gray + 0.5f);
    }
    return true;
}

bool savePNG(const std::string& filename, const std::vector<unsigned char>& image, unsigned width, unsigned height) {
    unsigned error = lodepng::encode(filename, image, width, height, LCT_GREY, 8);
    if (error) {
        std::cerr << "Ошибка lodepng при сохранении " << filename << ": "
                  << lodepng_error_text(error) << std::endl;
        return false;
    }
    return true;
}

bool loadImage(const std::string& filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height) {
    if (endsWith(filename, ".pgm")) {
        return loadPGM(filename, image, width, height);
    } else if (endsWith(filename, ".png")) {
        return loadPNG(filename, image, width, height);
    } else {
        std::cerr << "Поддерживаются только .pgm и .png файлы." << std::endl;
        return false;
    }
}

bool saveImage(const std::string& filename, const std::vector<unsigned char>& image, unsigned width, unsigned height) {
    if (endsWith(filename, ".pgm")) {
        return savePGM(filename, image, width, height);
    } else if (endsWith(filename, ".png")) {
        return savePNG(filename, image, width, height);
    } else {
        std::cerr << "Поддерживаются только .pgm и .png для вывода." << std::endl;
        return false;
    }
}
