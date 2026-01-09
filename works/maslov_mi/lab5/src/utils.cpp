#include "utils.h"
#include <fstream>
#include <cmath>
#include <algorithm>

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


struct Anchor {
    float cx, cy, w, h;
};

std::vector<Anchor> generate_anchors(int input_h, int input_w) {
    
    std::vector<int> strides = {8, 16, 32, 64, 128};
    std::vector<int> sizes = {32, 64, 128, 256, 512}; 
    std::vector<float> scales = {1.0f, pow(2.0f, 1.0f/3.0f), pow(2.0f, 2.0f/3.0f)};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    
    int internal_w = 800; 
    int internal_h = 800;

    std::vector<Anchor> anchors;
    
    for (size_t i = 0; i < strides.size(); ++i) {
        int stride = strides[i];
        int base_size = sizes[i];
        
        int grid_w = (int)ceil(internal_w / (float)stride);
        int grid_h = (int)ceil(internal_h / (float)stride);
        
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;
                
                for (float ratio : aspect_ratios) {
                    for (float scale : scales) {
                        float s_size = base_size * scale;
                        float w = s_size * sqrt(1.0f / ratio);
                        float h = s_size * sqrt(ratio);
                        anchors.push_back({cx, cy, w, h});
                    }
                }
            }
        }
    }
    return anchors;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

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
    float conf_thresh,
    float nms_thresh
) {
    // 1. Copy data 
    size_t cls_count = batch_size * num_anchors * num_classes;
    size_t bbox_count = batch_size * num_anchors * 4;
    
    checkCudaErrors(cudaMemcpyAsync(cpu_cls_logits, gpu_cls_logits, 
                              cls_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(cpu_bbox_regression, gpu_bbox_regression, 
                              bbox_count * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // 2. Generate Anchors
    std::vector<Anchor> anchors = generate_anchors(input_h, input_w);
    
    // Safety check
    if (anchors.size() != num_anchors) {
        std::cerr << "[ERROR] Mismatch anchor count! Generated: " << anchors.size() 
                  << " Expected: " << num_anchors << std::endl;
    }
    int process_anchors = std::min((int)anchors.size(), num_anchors);

    std::vector<std::vector<Detection>> batch_results(batch_size);
    
    float scale_x = (float)input_w / 800.0f;
    float scale_y = (float)input_h / 800.0f;

    for (int b = 0; b < batch_size; ++b) {
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> labels;
        
        for (int i = 0; i < process_anchors; ++i) {
            // Find max score among classes
            float max_score = -1.0f;
            int max_label = -1;
            
            int cls_offset = b * num_anchors * num_classes + i * num_classes;
            
            for (int c = 0; c < num_classes; ++c) {
                float score = sigmoid(cpu_cls_logits[cls_offset + c]);
                if (score > max_score) {
                    max_score = score;
                    max_label = c;
                }
            }
            
            if (max_score < conf_thresh) continue;
            
            // Decode box
            int bbox_offset = b * num_anchors * 4 + i * 4;
            float dx = cpu_bbox_regression[bbox_offset + 0];
            float dy = cpu_bbox_regression[bbox_offset + 1];
            float dw = cpu_bbox_regression[bbox_offset + 2];
            float dh = cpu_bbox_regression[bbox_offset + 3];
            
            Anchor& a = anchors[i];
            
            // Standard box decoding
            float pred_ctr_x = dx * a.w + a.cx;
            float pred_ctr_y = dy * a.h + a.cy;
            float pred_w = exp(dw) * a.w;
            float pred_h = exp(dh) * a.h;
            
            float x1 = pred_ctr_x - 0.5f * pred_w;
            float y1 = pred_ctr_y - 0.5f * pred_h;
            
            // Scale back to input resolution
            x1 *= scale_x;
            y1 *= scale_y;
            pred_w *= scale_x;
            pred_h *= scale_y;
            
            boxes.push_back(cv::Rect2d(x1, y1, pred_w, pred_h));
            scores.push_back(max_score);
            labels.push_back(max_label);
        }
        
        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thresh, nms_thresh, indices);
        
        for (int idx : indices) {
            Detection det;
            det.x1 = boxes[idx].x;
            det.y1 = boxes[idx].y;
            det.x2 = boxes[idx].x + boxes[idx].width;
            det.y2 = boxes[idx].y + boxes[idx].height;
            det.score = scores[idx];
            det.label = labels[idx];
            batch_results[b].push_back(det);
        }
    }
    
    return batch_results;
}

} // namespace utils
