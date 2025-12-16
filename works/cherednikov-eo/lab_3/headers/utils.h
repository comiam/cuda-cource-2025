#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

bool loadImage(const std::string& filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height);

bool saveImage(const std::string& filename, const std::vector<unsigned char>& image, unsigned width, unsigned height);

bool endsWith(const std::string& str, const std::string& suffix);