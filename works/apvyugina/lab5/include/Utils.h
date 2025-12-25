#include <stdint.h>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <regex>
#include <experimental/iterator>
#include <opencv2/opencv.hpp>

#ifndef UTILS
#define UTILS


struct Params
{
    std::string onnxFileName;
    std::string engineFileName;
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    bool int8{true};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    bool bf16{false};                  //!< Allow running the network in BF16 mode.
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;

    std::string saveEngine;
    std::string loadEngine;

    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t inputNChannels;

    uint32_t outputLength;
    uint32_t outputItemSize;

    std::string calibrationDataPath;  // Path to calibration images
    std::string calibrationCacheFile; // Optional: cache file path
};


class BBox{
    public:
        uint32_t x0;
        uint32_t y0;
        uint32_t x1;
        uint32_t y1;
        BBox(uint32_t x0_, uint32_t y0_, uint32_t x1_, uint32_t y1_):
            x0(x0_), y0(y0_), x1(x1_), y1(y1_) {
        }

};

class Detection{
    public: 
        int classId;
        float score;
        BBox bbox;

        Detection(BBox bbox_, float score_, int classId_):
            bbox(bbox_),
            score(score_),
            classId(classId_){
        }
    
};

class Utility{
    public:
        static cv::Mat processMat(Params p, cv::Mat image);
        static cv::Mat drawResult(
            cv::Mat img, 
            std::vector<Detection> detections
        );
        static std::vector<std::vector<Detection>> processOutput(
            float* output, 
            int numImages, 
            Params params, 
            float confThreshold
        );
        static Params createDefaultParams(const char* onnxFileName, const bool int8);
};


#endif