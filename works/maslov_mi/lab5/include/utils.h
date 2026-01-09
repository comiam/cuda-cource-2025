#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, char const* const func, const char* const file, int const line);

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int label;
};

namespace utils {
    // Загрузка названий классов из файла
    std::vector<std::string> loadLabels(const std::string& path);

    void preprocessBatch(const std::vector<cv::Mat>& frames, float* gpu_input, int width, int height, cudaStream_t stream);
    
    // Теперь принимает список классов для красивого вывода
    void drawDetections(std::vector<cv::Mat>& frames, 
                       const std::vector<std::vector<Detection>>& batch_detections,
                       const std::vector<std::string>& classNames);
}
