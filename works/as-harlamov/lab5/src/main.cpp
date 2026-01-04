#include "retinanet.h"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <labels_path> <input_video> [output_video] [conf_threshold]" << std::endl;
        return 1;
    }
    
    try {
        const std::string engine_path = argv[1];
        const std::string labels_path = argv[2];
        const std::string input_video = argv[3];
        const std::string output_video = argc > 4 ? argv[4] : "output.mp4";
        const float conf_threshold = argc > 5 ? std::stof(argv[5]) : 0.5f;
        
        RetinaNet model(engine_path, labels_path, conf_threshold);
    
        cv::VideoCapture cap(input_video);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video: " << input_video << std::endl;
            return 1;
        }
        
        const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "Cannot create video writer" << std::endl;
            return 1;
        }
        
        cv::Mat frame;
        int frame_count = 0;
        int total_detections = 0;
        
        const auto start_time = std::chrono::high_resolution_clock::now();
        
        while (cap.read(frame)) {
            if (frame.empty()) break;
            
            std::vector<Detection> detections = model.infer(frame);
            model.drawDetections(frame, detections);
            
            writer.write(frame);
            
            frame_count++;
            total_detections += detections.size();
            
            if (frame_count % 30 == 0) {
                std::cout << "Processed " << frame_count << " / " << total_frames << " frames" << std::endl;
            }
        }
        
        const auto end_time = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        const double processing_time = duration.count() / 1000.0;
        
        cap.release();
        writer.release();
        
        std::cout << "Video: " << input_video << " (" << width << "x" << height << ", " << fps << " FPS)" << std::endl;
        std::cout << "Processing time: " << processing_time << " sec" << std::endl;
        std::cout << "Detections: " << total_detections << std::endl;
        std::cout << "Output: " << output_video << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

