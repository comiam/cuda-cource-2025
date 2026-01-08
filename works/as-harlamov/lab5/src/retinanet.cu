#include "retinanet.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace {
    constexpr int BLOCK_SIZE_X = 16;
    constexpr int BLOCK_SIZE_Y = 16;
    constexpr int MAX_DETECTIONS = 1000;
    constexpr int NUM_CHANNELS = 3;
    constexpr int MAX_FRAME_WIDTH = 1920;
    constexpr int MAX_FRAME_HEIGHT = 1080;
    constexpr float PIXEL_NORMALIZATION = 255.0f;
}

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

__device__ __forceinline__ int clamp(int value, int min_val, int max_val) {
    return (value < min_val) ? min_val : (value >= max_val) ? (max_val - 1) : value;
}

__global__ void preprocess_kernel(
    const unsigned char* src, 
    float* dst,
    int src_width, 
    int src_height,
    int dst_width, 
    int dst_height,
    int src_pitch,
    int dst_c
) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    const float scale_x = static_cast<float>(src_width) / dst_width;
    const float scale_y = static_cast<float>(src_height) / dst_height;
    
    const float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    const float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
    
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    const float wx = src_x - x0;
    const float wy = src_y - y0;
    
    x0 = clamp(x0, 0, src_width);
    y0 = clamp(y0, 0, src_height);
    x1 = clamp(x1, 0, src_width);
    y1 = clamp(y1, 0, src_height);
    
    // BGR->RGB conversion: BGR channels (B=0, G=1, R=2) -> RGB output (R=0, G=1, B=2)
    const int bgr_channel = (dst_c == 0) ? 2 : (dst_c == 1) ? 1 : 0;
    
    const unsigned char p00 = src[y0 * src_pitch + x0 * 3 + bgr_channel];
    const unsigned char p10 = src[y0 * src_pitch + x1 * 3 + bgr_channel];
    const unsigned char p01 = src[y1 * src_pitch + x0 * 3 + bgr_channel];
    const unsigned char p11 = src[y1 * src_pitch + x1 * 3 + bgr_channel];
    
    const float interpolated = (1.0f - wx) * (1.0f - wy) * p00 +
                              wx * (1.0f - wy) * p10 +
                              (1.0f - wx) * wy * p01 +
                              wx * wy * p11;
    
    const int output_idx = dst_c * dst_height * dst_width + dst_y * dst_width + dst_x;
    dst[output_idx] = interpolated / PIXEL_NORMALIZATION;
}

RetinaNet::RetinaNet(const std::string& engine_path, const std::string& labels_path, float conf_threshold)
    : runtime(nullptr), engine(nullptr), context(nullptr),
      d_input(nullptr), d_boxes(nullptr), d_scores(nullptr), d_labels(nullptr), d_frame(nullptr),
      h_frame_pinned(nullptr), h_boxes_pinned(nullptr), h_scores_pinned(nullptr), h_labels_pinned(nullptr),
      stream_preprocess(nullptr), stream_inference(nullptr), stream_postprocess(nullptr),
      input_h(640), input_w(640), conf_threshold(conf_threshold) {
    loadEngine(engine_path);
    loadLabels(labels_path);
    allocateBuffers();
}

RetinaNet::~RetinaNet() {
    if (stream_preprocess) CUDA_CHECK(cudaStreamSynchronize(stream_preprocess));
    if (stream_inference) CUDA_CHECK(cudaStreamSynchronize(stream_inference));
    if (stream_postprocess) CUDA_CHECK(cudaStreamSynchronize(stream_postprocess));
    
    if (d_input) CUDA_CHECK(cudaFree(d_input));
    if (d_boxes) CUDA_CHECK(cudaFree(d_boxes));
    if (d_scores) CUDA_CHECK(cudaFree(d_scores));
    if (d_labels) CUDA_CHECK(cudaFree(d_labels));
    if (d_frame) CUDA_CHECK(cudaFree(d_frame));
    
    if (h_frame_pinned) CUDA_CHECK(cudaFreeHost(h_frame_pinned));
    if (h_boxes_pinned) CUDA_CHECK(cudaFreeHost(h_boxes_pinned));
    if (h_scores_pinned) CUDA_CHECK(cudaFreeHost(h_scores_pinned));
    if (h_labels_pinned) CUDA_CHECK(cudaFreeHost(h_labels_pinned));
    
    if (stream_preprocess) CUDA_CHECK(cudaStreamDestroy(stream_preprocess));
    if (stream_inference) CUDA_CHECK(cudaStreamDestroy(stream_inference));
    if (stream_postprocess) CUDA_CHECK(cudaStreamDestroy(stream_postprocess));
    
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
}

void RetinaNet::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open engine file: " + engine_path);
    }
    
    file.seekg(0, std::ios::end);
    const size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read engine file: " + engine_path);
    }
    
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    
    engine = runtime->deserializeCudaEngine(buffer.data(), size);
    if (!engine) {
        runtime->destroy();
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    
    context = engine->createExecutionContext();
    if (!context) {
        engine->destroy();
        runtime->destroy();
        throw std::runtime_error("Failed to create execution context");
    }
}

void RetinaNet::loadLabels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file) {
        throw std::runtime_error("Cannot open labels file: " + labels_path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
}

void RetinaNet::allocateBuffers() {
    input_size = NUM_CHANNELS * input_h * input_w * sizeof(float);
    boxes_size = MAX_DETECTIONS * 4 * sizeof(float);
    scores_size = MAX_DETECTIONS * sizeof(float);
    labels_size = MAX_DETECTIONS * sizeof(int);
    max_frame_size = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * NUM_CHANNELS * sizeof(unsigned char);
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_boxes, boxes_size));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_labels, labels_size));
    CUDA_CHECK(cudaMalloc(&d_frame, max_frame_size));
    
    CUDA_CHECK(cudaMallocHost(&h_frame_pinned, max_frame_size));
    CUDA_CHECK(cudaMallocHost(&h_boxes_pinned, boxes_size));
    CUDA_CHECK(cudaMallocHost(&h_scores_pinned, scores_size));
    CUDA_CHECK(cudaMallocHost(&h_labels_pinned, labels_size));
    
    CUDA_CHECK(cudaStreamCreate(&stream_preprocess));
    CUDA_CHECK(cudaStreamCreate(&stream_inference));
    CUDA_CHECK(cudaStreamCreate(&stream_postprocess));
}

void RetinaNet::preprocess(const cv::Mat& frame, cudaStream_t stream) {
    const int src_width = frame.cols;
    const int src_height = frame.rows;
    const int src_pitch = frame.step;
    
    const size_t frame_size = src_height * src_pitch;
    
    std::memcpy(h_frame_pinned, frame.data, frame_size);
    
    CUDA_CHECK(cudaMemcpyAsync(d_frame, h_frame_pinned, frame_size, 
                                cudaMemcpyHostToDevice, stream));
    
    const dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 grid((input_w + block.x - 1) / block.x, 
                    (input_h + block.y - 1) / block.y);
    
    // Output format: CHW (Channel-Height-Width)
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        preprocess_kernel<<<grid, block, 0, stream>>>(
            static_cast<const unsigned char*>(d_frame),
            static_cast<float*>(d_input),
            src_width,
            src_height,
            input_w,
            input_h,
            src_pitch,
            c
        );
    }
}

void RetinaNet::postprocess(float* boxes, float* scores, int* labels, int num_detections, std::vector<Detection>& detections) {
    detections.clear();
    detections.reserve(num_detections);
    
    for (int i = 0; i < num_detections; ++i) {
        if (scores[i] < conf_threshold) continue;
        
        Detection det;
        det.x1 = boxes[i * 4 + 0];
        det.y1 = boxes[i * 4 + 1];
        det.x2 = boxes[i * 4 + 2];
        det.y2 = boxes[i * 4 + 3];
        det.score = scores[i];
        det.label = labels[i];
        
        detections.push_back(det);
    }
}

std::vector<Detection> RetinaNet::infer(const cv::Mat& frame) {
    preprocess(frame, stream_preprocess);
    
    cudaEvent_t preprocess_done;
    CUDA_CHECK(cudaEventCreate(&preprocess_done));
    CUDA_CHECK(cudaEventRecord(preprocess_done, stream_preprocess));
    CUDA_CHECK(cudaStreamWaitEvent(stream_inference, preprocess_done, 0));
    
    void* bindings[] = {d_input, d_boxes, d_scores, d_labels};
    if (!context->enqueueV2(bindings, stream_inference, nullptr)) {
        CUDA_CHECK(cudaEventDestroy(preprocess_done));
        throw std::runtime_error("Failed to execute TensorRT inference");
    }
    
    cudaEvent_t inference_done;
    CUDA_CHECK(cudaEventCreate(&inference_done));
    CUDA_CHECK(cudaEventRecord(inference_done, stream_inference));
    CUDA_CHECK(cudaStreamWaitEvent(stream_postprocess, inference_done, 0));
    
    CUDA_CHECK(cudaMemcpyAsync(h_boxes_pinned, d_boxes, boxes_size, 
                                cudaMemcpyDeviceToHost, stream_postprocess));
    CUDA_CHECK(cudaMemcpyAsync(h_scores_pinned, d_scores, scores_size, 
                                cudaMemcpyDeviceToHost, stream_postprocess));
    CUDA_CHECK(cudaMemcpyAsync(h_labels_pinned, d_labels, labels_size, 
                                cudaMemcpyDeviceToHost, stream_postprocess));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_postprocess));
    
    CUDA_CHECK(cudaEventDestroy(preprocess_done));
    CUDA_CHECK(cudaEventDestroy(inference_done));
    
    int num_detections = 0;
    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        if (h_scores_pinned[i] > 0.0f) {
            num_detections++;
        }
    }
    
    std::vector<Detection> detections;
    postprocess(h_boxes_pinned, h_scores_pinned, h_labels_pinned, num_detections, detections);
    
    return detections;
}

void RetinaNet::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    const float scale_x = static_cast<float>(frame.cols) / input_w;
    const float scale_y = static_cast<float>(frame.rows) / input_h;
    const int frame_width = frame.cols;
    const int frame_height = frame.rows;
    
    for (const auto& det : detections) {
        int x1 = static_cast<int>(det.x1 * scale_x);
        int y1 = static_cast<int>(det.y1 * scale_y);
        int x2 = static_cast<int>(det.x2 * scale_x);
        int y2 = static_cast<int>(det.y2 * scale_y);
        
        x1 = std::max(0, std::min(x1, frame_width - 1));
        y1 = std::max(0, std::min(y1, frame_height - 1));
        x2 = std::max(0, std::min(x2, frame_width - 1));
        y2 = std::max(0, std::min(y2, frame_height - 1));
        
        const cv::Scalar color(0, 255, 0);
        const int thickness = 2;
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
        
        std::string score_str = std::to_string(det.score);
        const size_t dot_pos = score_str.find('.');
        if (dot_pos != std::string::npos) {
            score_str = score_str.substr(0, dot_pos + 3);
        }
        
        const std::string label_name = (det.label >= 0 && det.label < static_cast<int>(labels.size())) 
                                      ? labels[det.label] 
                                      : "unknown";
        const std::string label_text = label_name + " " + score_str;
        
        cv::putText(frame, label_text, cv::Point(x1, y1 - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, thickness);
    }
}

