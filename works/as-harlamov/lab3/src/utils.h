#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

bool loadImage(const std::string& filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height);

bool saveImage(const std::string& filename, const std::vector<unsigned char>& image, unsigned width, unsigned height);

bool endsWith(const std::string& str, const std::string& suffix);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#endif // UTILS_H