#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include "engine.h"
#include "utils.h"

namespace fs = std::filesystem;

const int BATCH_SIZE = 16;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "<onnx_path> video <input_mp4> <use_int8_bool>" << std::endl;
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

    ModelOptions opts;
    opts.onnx_path = onnx_path;
    opts.use_int8 = use_int8;
    opts.max_batch_size = BATCH_SIZE;
    opts.input_width = frame_width;
    opts.input_height = frame_height;
    opts.engine_path = "model_" + std::to_string(frame_width) + "x" + std::to_string(frame_height) + ".engine";

    InferenceEngine engine(opts);

    fs::path onnx_p(onnx_path);
    fs::path labels_p = onnx_p.parent_path() / "labels.txt";
    std::vector<std::string> classNames = utils::loadLabels(labels_p.string());
    if (classNames.empty()) classNames = utils::loadLabels("labels.txt");

    if (!engine.load()) {
        std::cout << "Building engine for " << frame_width << "x" << frame_height << "..." << std::endl;
        if (!engine.build()) return -1;
        if (!engine.load()) return -1;
    }

    void* buffers[4]; 
    const int MAX_DET = 300; 
    
    size_t input_size = BATCH_SIZE * 3 * frame_height * frame_width * sizeof(float);
    size_t boxes_size = BATCH_SIZE * MAX_DET * 4 * sizeof(float);
    size_t scores_size = BATCH_SIZE * MAX_DET * sizeof(float);
    size_t labels_size = BATCH_SIZE * MAX_DET * sizeof(int64_t);

    checkCudaErrors(cudaMalloc(&buffers[0], input_size));
    checkCudaErrors(cudaMalloc(&buffers[1], boxes_size));
    checkCudaErrors(cudaMalloc(&buffers[2], scores_size));
    checkCudaErrors(cudaMalloc(&buffers[3], labels_size));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    cv::VideoWriter writer("output_annotated.mp4", 
                           cv::VideoWriter::fourcc('m','p','4','v'), 
                           fps, cv::Size(frame_width, frame_height));

    std::vector<cv::Mat> batch_frames;
    batch_frames.reserve(BATCH_SIZE);
    
    std::vector<float> cpu_boxes(BATCH_SIZE * MAX_DET * 4);
    std::vector<float> cpu_scores(BATCH_SIZE * MAX_DET);
    std::vector<int64_t> cpu_labels(BATCH_SIZE * MAX_DET);

    std::cout << "Start processing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    int total_frames = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        batch_frames.push_back(frame);

        if (batch_frames.size() == BATCH_SIZE) {
            utils::preprocessBatch(batch_frames, (float*)buffers[0], frame_width, frame_height, stream);
            engine.run(buffers, stream);

            checkCudaErrors(cudaMemcpyAsync(cpu_boxes.data(), buffers[1], boxes_size, cudaMemcpyDeviceToHost, stream));
            checkCudaErrors(cudaMemcpyAsync(cpu_scores.data(), buffers[2], scores_size, cudaMemcpyDeviceToHost, stream));
            checkCudaErrors(cudaMemcpyAsync(cpu_labels.data(), buffers[3], labels_size, cudaMemcpyDeviceToHost, stream));
            
            cudaStreamSynchronize(stream);

            std::vector<std::vector<Detection>> batch_dets(BATCH_SIZE);
            
            for(int b=0; b < BATCH_SIZE; ++b) {
                for(int i=0; i < MAX_DET; ++i) {
                    float score = cpu_scores[b * MAX_DET + i];
                    if (score < 0.4f) continue;

                    Detection det;
                    det.score = score;
                    det.label = (int)cpu_labels[b * MAX_DET + i];
                    
                    int box_idx = (b * MAX_DET + i) * 4;
                    det.x1 = cpu_boxes[box_idx + 0];
                    det.y1 = cpu_boxes[box_idx + 1];
                    det.x2 = cpu_boxes[box_idx + 2];
                    det.y2 = cpu_boxes[box_idx + 3];
                    
                    batch_dets[b].push_back(det);
                }
            }

            utils::drawDetections(batch_frames, batch_dets, classNames);

            for (const auto& f : batch_frames) {
                writer.write(f);
            }
            
            total_frames += BATCH_SIZE;
            if (total_frames % 20 == 0) {
                 std::cout << "\rFrames: " << total_frames << std::flush;
            }

            batch_frames.clear();
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaFree(buffers[3]);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "\nDone. Saved to output_annotated.mp4" << std::endl;
    std::cout << "Avg FPS: " << total_frames / diff.count() << std::endl;

    return 0;
}
