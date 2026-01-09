#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, char const* const func, const char* const file, int const line);

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int label;
};

namespace utils {
    std::vector<std::string> loadLabels(const std::string& path);

    void preprocessBatch(const std::vector<cv::Mat>& frames, float* gpu_input, int width, int height, cudaStream_t stream);
    
    void drawDetections(std::vector<cv::Mat>& frames, 
                       const std::vector<std::vector<Detection>>& batch_detections,
                       const std::vector<std::string>& classNames);

    std::vector<std::vector<Detection>> postProcess(
        float* gpu_cls_logits, 
        float* gpu_bbox_regression, 
        float* cpu_cls_logits, 
        float* cpu_bbox_regression,
        int batch_size, 
        int num_anchors, 
        int num_classes,
        int input_w,
        int input_h,
        cudaStream_t stream,
        float conf_thresh = 0.4f,
        float nms_thresh = 0.4f
    );
}
