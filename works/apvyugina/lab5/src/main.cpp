#include "DetectionModelTRT.h"
#include "Timers.h"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <vector>

#define BATCH_SIZE 16

using namespace std;


void detectOnRandomImages(
    const string& inputImageFolder, 
    const string& outputImageFolder,
    DetectionModelTRT& engine,
    Params p,
    float confThreshold,
    int batchSize
){
    
    const filesystem::path img_folder{inputImageFolder};

    vector<cv::Mat> preprocessedImgList;
    for (auto const& dir_entry : filesystem::directory_iterator{img_folder}){
        if (filesystem::is_directory(dir_entry)) continue;
        
        cv::Mat image = cv::imread(dir_entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) continue;
        preprocessedImgList.push_back(Utility::processMat(p, image));

    }

    int numberOfImages = preprocessedImgList.size();
    cout << "Total number of Images in directory: " << numberOfImages << endl;

    Timer timer;
    float* rawOutput = nullptr;

    timer.tic();
    engine.detect(preprocessedImgList, rawOutput);
    double diff = timer.toc();
    cout << "Batch size=" <<numberOfImages << " took " << diff  << " ms, "  <<
        diff/numberOfImages << " ms/img" << endl;

    vector<vector<Detection>> resultList = Utility::processOutput(rawOutput, batchSize, p, confThreshold);
        
    for(int i = 0; i < batchSize; ++i){
        auto img = preprocessedImgList[i];
        auto result = resultList[i];

        string filename = outputImageFolder + to_string(i) + ".png";
        cv::Mat resultImage = Utility::drawResult(img, result);
        cv::imwrite(filename, resultImage);
    }
}


void detectOnVideo(
    const string& inputVideoPath, 
    const string& outputVideoPath, 
    DetectionModelTRT& engine, 
    Params p,
    float confThreshold
) {
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) throw runtime_error("Could not open video file.");

    cv::Mat frame;
    vector<cv::Mat> framesBatch;

    cv::Size size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer(outputVideoPath, 
                       cv::VideoWriter::fourcc('M','J','P','G'), 
                       cap.get(cv::CAP_PROP_FPS), size);
    
    int currentBS;
    vector<cv::Mat> preprocessedFrames;
    vector<cv::Mat> processedFrames;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        framesBatch.emplace_back(frame.clone());
        if (framesBatch.size() >= BATCH_SIZE || !cap.read(frame)) {    
            currentBS = framesBatch.size();

            preprocessedFrames.reserve(currentBS);
            processedFrames.reserve(currentBS);
            cv::Mat preprocessed_frame;
            cv::Mat convertedFrame;
            for (const auto& frame: framesBatch){
                preprocessed_frame = Utility::processMat(p, frame);
                preprocessedFrames.push_back(preprocessed_frame);
            }

            float* rawOutput = nullptr;
            engine.detect(preprocessedFrames, rawOutput);

            vector<vector<Detection>> resultList = Utility::processOutput(
                rawOutput, preprocessedFrames.size(), p, confThreshold
            );
        
            for (size_t i = 0; i < currentBS && i < resultList.size(); ++i) {
                cv::Mat processed_frame = Utility::drawResult(preprocessedFrames[i], resultList[i]);
                cv::Mat resized_frame;
                cv::resize(processed_frame, resized_frame, size);
                processedFrames.push_back(resized_frame);
            }

            for (const auto& img : processedFrames) {
                writer.write(img);
            }

            framesBatch.clear();
            preprocessedFrames.clear();
            processedFrames.clear();
            delete[] rawOutput;
        }
    }

    writer.release();
}


int main(int argc, char** argv)
{   
    if (argc != 5){
        cerr << "Usage: " << argv[0] << " onnxFileName images|video /path/to/directory_or_video int8_optimization\n";
        return 1;  
    }

    char* onnxFileName = argv[1];
    filesystem::path onnxFilePath(onnxFileName);

    const std::string type(argv[2]);
    const std::string path(argv[3]);
    const std::string int8Arg(argv[4]);

    // Проверяем первый аргумент на валидность
    if ((type != "images") && (type != "video")) {
        cerr << "Error: First argument must be 'images' or 'video'\n";
        return 1;
    }

    if (!filesystem::exists(path)) {
        cerr << "Error: Path '" << path << "' does not exist.\n";
        return 1;
    }

    // Parse int8 optimization argument
    bool int8Optimization = false;
    if (int8Arg == "true" || int8Arg == "1" || int8Arg == "yes") {
        int8Optimization = true;
    } else if (int8Arg == "false" || int8Arg == "0" || int8Arg == "no") {
        int8Optimization = false;
    } else {
        cerr << "Error: Fourth argument must be 'true'/'false', '1'/'0', or 'yes'/'no'\n";
        return 1;
    }

    Params params = Utility::createDefaultParams(onnxFileName, int8Optimization);
    params.int8 = int8Optimization;
    cout << "Building Engine with params:\n"
          << "- ONNX file path: " << params.onnxFileName << "\n"
          << "- Engine file name: " << params.engineFileName << "\n"
          << "- Optimizations: INT8=" << boolalpha << params.int8 << " "
          << "FP16=" << boolalpha << params.fp16 << " "
          << "BF16=" << boolalpha << params.bf16 << "\n"
          << "- Input (HxWxC): " << params.inputHeight << "x" << params.inputWidth << "x" << params.inputNChannels << "\n"
          << "- Output: " << params.outputLength << "x" << params.outputItemSize << "\n"
          << "- Calibration data path: " << params.calibrationDataPath << "\n"
          << "- Calibration cache file: " << params.calibrationCacheFile << "\n";
    DetectionModelTRT Engine(params);

    bool status = Engine.prepareEngine();
    if (!status) {
        cerr << "Failed to prepare engine" << endl;
        return 1;
    }
    
    if (type == "images"){
        detectOnRandomImages(
            path,
            "results/trt/",
            Engine,
            params,
            0.6,
            10
        );
    }
    else if (type == "video"){
        detectOnVideo(
            path,
            "videos/processed.avi",
            Engine,
            params, 
            0.5
        );
    }

    return 0;
}