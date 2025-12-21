#include "Utils.h"

using namespace std;

vector<cv::Mat> Utility::processInput(Params p, const filesystem::path img_folder){
    vector<cv::Mat> preprocessedImgList;
    for (auto const& dir_entry : filesystem::directory_iterator{img_folder}){
        if (filesystem::is_directory(dir_entry)) continue;
        
        // Load image as BGR (OpenCV default)
        cv::Mat image = cv::imread(dir_entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) continue;
        assert(image.channels() == 3);

        // Convert BGR to RGB
        cv::Mat image_rgb;
        cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

        // Resize FIRST on uint8 (matches CImg behavior - resize before normalization)
        // Use INTER_LINEAR to match CImg's default resize interpolation
        cv::Mat resized;
        cv::resize(image_rgb, resized, cv::Size(p.inputWidth, p.inputHeight), 0, 0, cv::INTER_LINEAR);

        // Normalize to [0, 1] float range AFTER resize (matches CImg normalize(0.0F, 1.0F))
        cv::Mat normalized;
        resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

        assert(normalized.rows == p.inputHeight && normalized.cols == p.inputWidth && normalized.channels() == p.inputNChannels);
        
        // Ensure contiguous memory layout
        cv::Mat final_img = normalized.clone();

        preprocessedImgList.push_back(final_img);

    }
    return preprocessedImgList;
};


void Utility::drawResult(
    cv::Mat img, 
    vector<Detection> detections, 
    const char* file_name
){
    // Convert from [0,1] float RGB to [0,255] uchar BGR for display
    cv::Mat img_display;
    img.convertTo(img_display, CV_8U, 255.0);
    cv::Mat img_bgr;
    cv::cvtColor(img_display, img_bgr, cv::COLOR_RGB2BGR);
    
    // Draw bounding boxes and labels
    for (Detection const& det : detections){
        cv::Point pt1(det.bbox.x0, det.bbox.y0);
        cv::Point pt2(det.bbox.x1, det.bbox.y1);
        cv::Scalar color(0, 0, 255); // Red in BGR
        cv::rectangle(img_bgr, pt1, pt2, color, 2);
        
        // Draw class ID text
        string label = to_string(det.classId);
        cv::Point textPos(det.bbox.x0, det.bbox.y0 - 5);
        cv::Scalar textColor(255, 255, 255); // White
        cv::Scalar bgColor(0, 0, 0); // Black background
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(img_bgr, textPos + cv::Point(0, baseline), 
                     textPos + cv::Point(textSize.width, -textSize.height), bgColor, -1);
        cv::putText(img_bgr, label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1);
    }

    cv::imwrite(file_name, img_bgr);
};



vector<vector<Detection>> Utility::processOutput(float* output, int numImages, Params params, float confThreshold)
{
    vector<vector<Detection>> resultList;
    
    uint32_t row_ptr;
    uint32_t floatsPerImage = params.outputItemSize * params.outputLength;
    for (int i=0; i < numImages; ++i){
        vector<Detection> result;
        for (int j=0; j < params.outputLength; ++j){
            if (j==0){
                for (int k=0; k < params.outputItemSize; ++k){
                    cout << output[i * floatsPerImage + j * params.outputItemSize + k] << " ";
                }
                cout << endl;
            }
            if (output[i * floatsPerImage + j * params.outputItemSize + 4] < 0.5) continue;
            row_ptr = i * floatsPerImage + j * params.outputItemSize;
            // Scale normalized coordinates (0-1) to pixel coordinates
            result.push_back(
                Detection(
                    BBox(output[row_ptr], output[row_ptr+1], output[row_ptr+2], output[row_ptr+3]),
                    output[row_ptr+4], 
                    static_cast<int>(output[row_ptr+5])
                )
            );
            
            for (int k=0; k < params.outputItemSize; ++k){
                cout << output[row_ptr + k] << " ";
            }
            cout << endl;
            
            
        }
        resultList.push_back(result);
    }
    
    return resultList;
}


Params Utility::createDefaultParams(const char* onnxFileName) {
    Params params;
    
    params.inputTensorNames.push_back("images");
    params.outputTensorNames.push_back("output0");   
    
    params.dlaCore = -1; // not supported on the server
    params.int8 = false;
    params.fp16 = false;
    params.bf16 = false;
    
    params.inputHeight = 640;
    params.inputWidth = 640;
    params.inputNChannels = 3;
    
    params.outputLength = 300;
    params.outputItemSize = 6;
    
    params.calibrationDataPath = "assets/"; 
    params.calibrationCacheFile = "models/calibration.cache";
    
    filesystem::path onnxFilePath(onnxFileName);
    string baseName = onnxFilePath.stem().string(); 
    auto parentDir = onnxFilePath.parent_path();    

    string suffix;
    if (params.int8) suffix += ".int8";
    else if (params.bf16) suffix += ".bf16";
    else if (params.fp16) suffix += ".fp16";

    params.onnxFileName = onnxFileName;
    params.engineFileName = (parentDir / (baseName + suffix + ".engine")).string();

    return params;
}