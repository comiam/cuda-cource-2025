#ifndef RETINANET_H
#define RETINANET_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int label;
};

class RetinaNet {
public:
    RetinaNet(const std::string& engine_path, const std::string& labels_path, float conf_threshold = 0.5f);
    ~RetinaNet();
    
    std::vector<Detection> infer(const cv::Mat& frame);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    
    void* d_input;
    void* d_boxes;
    void* d_scores;
    void* d_labels;
    void* d_frame;
    
    unsigned char* h_frame_pinned;
    float* h_boxes_pinned;
    float* h_scores_pinned;
    int* h_labels_pinned;
    
    cudaStream_t stream_preprocess;
    cudaStream_t stream_inference;
    cudaStream_t stream_postprocess;
    
    size_t input_size;
    size_t boxes_size;
    size_t scores_size;
    size_t labels_size;
    size_t max_frame_size;
    
    int input_h;
    int input_w;
    float conf_threshold;
    std::vector<std::string> labels;
    
    void loadEngine(const std::string& engine_path);
    void loadLabels(const std::string& labels_path);
    void allocateBuffers();
    void preprocess(const cv::Mat& frame, cudaStream_t stream);
    void postprocess(float* boxes, float* scores, int* labels, int num_detections, std::vector<Detection>& detections);
};

#endif

