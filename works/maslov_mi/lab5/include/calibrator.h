#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(int batchSize, int inputW, int inputH, 
                          const std::string& calibrationDataPath, 
                          const std::string& calibTableName,
                          const std::string& inputBlobName, 
                          bool readCache = true);

    virtual ~Int8EntropyCalibrator2();

    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int mBatchSize;
    int mInputW;
    int mInputH;
    std::string mInputBlobName;
    std::string mCalibTableName;
    bool mReadCache;
    
    std::vector<char> mCalibrationCache;
    std::vector<float> mBatchData;
    void* mDeviceInput{nullptr};
    
    // Video handling for calibration
    cv::VideoCapture mCap;
    int mTotalFrames;
    int mCurrentFrame;
    int mMaxCalibrationBatches;
    int mCurrentBatch;
};

