#include "utils.h"
#include <fstream>

void check(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error in " << file << ":" << line 
                  << " code=" << result << " func=\"" << func << "\" \n";
        exit(static_cast<unsigned int>(result));
    }
}

namespace utils {

std::vector<std::string> loadLabels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[WARN] labels.txt not found at " << path << ", using ID instead." << std::endl;
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

void preprocessBatch(const std::vector<cv::Mat>& frames, float* gpu_input, int width, int height, cudaStream_t stream) {
    cv::Mat blob;
    cv::dnn::blobFromImages(frames, blob, 1.0/255.0, cv::Size(width, height), cv::Scalar(0,0,0), true, false);
    
    size_t batch_size = frames.size();
    size_t image_size = 3 * width * height;
    
    checkCudaErrors(cudaMemcpyAsync(gpu_input, blob.ptr<float>(), 
                                   batch_size * image_size * sizeof(float), 
                                   cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));
}

void drawDetections(std::vector<cv::Mat>& frames, 
                   const std::vector<std::vector<Detection>>& batch_detections,
                   const std::vector<std::string>& classNames) {
    for (size_t i = 0; i < frames.size(); ++i) {
        for (const auto& det : batch_detections[i]) {
            cv::rectangle(frames[i], cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
            
            std::string label;
            if (det.label >= 0 && det.label < classNames.size()) {
                label = classNames[det.label];
            } else {
                label = "Class " + std::to_string(det.label);
            }
            label += ": " + std::to_string(int(det.score * 100)) + "%";
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frames[i], cv::Point(det.x1, det.y1 - labelSize.height - 5), 
                                     cv::Point(det.x1 + labelSize.width, det.y1), 
                                     cv::Scalar(0, 255, 0), cv::FILLED);
            
            cv::putText(frames[i], label, cv::Point(det.x1, det.y1 - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
}

} // namespace utils
