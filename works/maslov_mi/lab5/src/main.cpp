#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include "engine.h"
#include "utils.h"

namespace fs = std::filesystem;

const int BATCH_SIZE = 1;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./RetinaTRT <onnx_path> video <input_mp4> <use_int8_bool>" << std::endl;
        return 1;
    }

    std::string onnx_path = argv[1];
    std::string input_video_path = argv[3];
    bool use_int8 = (std::string(argv[4]) == "true");

    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    std::cout << "Video input: " << frame_width << "x" << frame_height << " @ " << fps << " FPS" << std::endl;

    const int MODEL_WIDTH = 640;
    const int MODEL_HEIGHT = 640;

    ModelOptions opts;
    opts.onnx_path = onnx_path;
    opts.use_int8 = use_int8;
    opts.max_batch_size = BATCH_SIZE;
    opts.input_width = MODEL_WIDTH;
    opts.input_height = MODEL_HEIGHT;
    opts.engine_path = "model_" + std::to_string(MODEL_WIDTH) + "x" + std::to_string(MODEL_HEIGHT) + ".engine";
    opts.calibration_data_path = input_video_path; // Use input video for calibration if needed

    InferenceEngine engine(opts);

    // Look for labels.txt in the same dir as ONNX or current dir
    fs::path onnx_p(onnx_path);
    fs::path labels_p = onnx_p.parent_path() / "labels.txt";
    std::vector<std::string> classNames = utils::loadLabels(labels_p.string());
    if (classNames.empty()) classNames = utils::loadLabels("labels.txt");
    // if (classNames.empty()) classNames = utils::loadLabels("../onnx_scripts/labels.txt"); // Fallback for build dir run

    if (!engine.load()) {
        std::cout << "Building engine for " << frame_width << "x" << frame_height << "..." << std::endl;
        if (!engine.build()) return -1;
        if (!engine.load()) return -1;
    }

    // For Raw RetinaNet V2: 1 input + 2 outputs
    const int NUM_BINDINGS = 3;
    void* buffers[NUM_BINDINGS];
    
    size_t input_size = BATCH_SIZE * 3 * MODEL_HEIGHT * MODEL_WIDTH * sizeof(float);
    checkCudaErrors(cudaMalloc(&buffers[0], input_size));

    size_t cls_size = BATCH_SIZE * 200000 * 91 * sizeof(float);
    size_t bbox_size = BATCH_SIZE * 200000 * 4 * sizeof(float);

    checkCudaErrors(cudaMalloc(&buffers[1], cls_size));
    checkCudaErrors(cudaMalloc(&buffers[2], bbox_size));

    // Allocate pinned host memory for output
    float* cpu_cls_buffer = nullptr;
    float* cpu_bbox_buffer = nullptr;
    checkCudaErrors(cudaMallocHost((void**)&cpu_cls_buffer, cls_size));
    checkCudaErrors(cudaMallocHost((void**)&cpu_bbox_buffer, bbox_size));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    cv::VideoWriter writer("output_annotated.mp4", 
                           cv::VideoWriter::fourcc('m','p','4','v'), 
                           fps, cv::Size(frame_width, frame_height));

    std::vector<cv::Mat> batch_frames;
    batch_frames.reserve(BATCH_SIZE);
    
    std::cout << "Start processing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    int total_frames = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        batch_frames.push_back(frame);

        if (batch_frames.size() == BATCH_SIZE) {
            utils::preprocessBatch(batch_frames, (float*)buffers[0], MODEL_WIDTH, MODEL_HEIGHT, stream);
            engine.run(buffers, stream);
            
            int num_anchors = 120087;
            int num_classes = 91;
            
            auto detections = utils::postProcess(
                (float*)buffers[1], 
                (float*)buffers[2], 
                cpu_cls_buffer,
                cpu_bbox_buffer,
                batch_frames.size(), 
                num_anchors, 
                num_classes,
                MODEL_WIDTH, 
                MODEL_HEIGHT,
                stream,
                0.5f,
                0.5f 
            );
            
            float scale_x = (float)frame_width / MODEL_WIDTH;
            float scale_y = (float)frame_height / MODEL_HEIGHT;

            for (auto& batch : detections) {
                for (auto& det : batch) {
                    det.x1 *= scale_x;
                    det.x2 *= scale_x;
                    det.y1 *= scale_y;
                    det.y2 *= scale_y;
                }
            }
            
            utils::drawDetections(batch_frames, detections, classNames);

            for (const auto& f : batch_frames) {
                writer.write(f);
            }
            
            total_frames += BATCH_SIZE;
            if (total_frames % 20 == 0) {
                 std::cout << "\rFrames processed: " << total_frames << std::flush;
            }

            batch_frames.clear();
        }
    }

    cudaStreamDestroy(stream);
    for(int i=0; i<NUM_BINDINGS; ++i) cudaFree(buffers[i]);
    cudaFreeHost(cpu_cls_buffer);
    cudaFreeHost(cpu_bbox_buffer);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "\nDone. Saved to output_annotated.mp4" << std::endl;
    std::cout << "Avg FPS: " << total_frames / diff.count() << std::endl;

    return 0;
}
