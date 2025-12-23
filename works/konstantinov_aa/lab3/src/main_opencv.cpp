// ============================================
// Sobel Filter with CUDA + OpenCV
// ============================================

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

// CUDA wrapper из sobel.cu
extern "C" int run_sobel_wrapper(const unsigned char* h_input, unsigned char* h_output, int width, int height);

const std::string DEFAULT_INPUT = "images/test_input.jpg";
const std::string DEFAULT_OUTPUT = "images/out_sobel.jpg";

int main(int argc, char* argv[]) {
    
    std::string inputFile = DEFAULT_INPUT;
    std::string outputFile = DEFAULT_OUTPUT;
    
    if (argc >= 2) {
        inputFile = argv[1];
    }
    if (argc >= 3) {
        outputFile = argv[2];
    }


    cv::Mat inputImage = cv::imread(inputFile, cv::IMREAD_COLOR);
    
    if (inputImage.empty()) {
        std::cerr << "Cannot load image: " << inputFile << std::endl;
        return -1;
    }
    
    int width = inputImage.cols;
    int height = inputImage.rows;
    std::cout << "Loaded: " << width << "x" << height << " (" << inputImage.channels() << " channels)" << std::endl;

    // Конвертация в Grayscale через OpenCV
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    
    cv::Mat outputImage(height, width, CV_8UC1);

    
    int status = run_sobel_wrapper(
        grayImage.data,      
        outputImage.data,    
        width, 
        height
    );

    if (status != 0) {
        std::cerr << "CUDA processing failed!" << std::endl;
        return -1;
    }
    std::cout << "CUDA kernel completed successfully" << std::endl;

   
    if (cv::imwrite(outputFile, outputImage)) {
        std::cout << "Saved result to: " << outputFile << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
        return -1;
    }

    
    return 0;
}
