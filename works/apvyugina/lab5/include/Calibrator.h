#include "NvInfer.h"
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(
        const std::string& calibrationDataPath,
        int batchSize,
        int inputHeight,
        int inputWidth,
        int inputChannels,
        const std::string& cacheFile = ""
    ) : mBatchSize(batchSize),
        mInputHeight(inputHeight),
        mInputWidth(inputWidth),
        mInputChannels(inputChannels),
        mCacheFile(cacheFile),
        mCurrentBatch(0),
        mDeviceInput(nullptr)
    {
        loadCalibrationImages(calibrationDataPath);
        
        if (mCalibrationImages.empty()) {
            throw std::runtime_error("No calibration images loaded from: " + calibrationDataPath);
        }
        
        // Allocate GPU memory for batch with error checking
        size_t inputSize = mBatchSize * mInputChannels * mInputHeight * mInputWidth * sizeof(float);
        cudaError_t err = cudaMalloc(&mDeviceInput, inputSize);
        if (err != cudaSuccess || mDeviceInput == nullptr) {
            throw std::runtime_error("Failed to allocate CUDA memory for calibrator: " + 
                std::string(cudaGetErrorString(err)));
        }
        
        // Initialize memory to zero
        cudaMemset(mDeviceInput, 0, inputSize);
    }

    ~Int8EntropyCalibrator2()
    {
        cudaFree(mDeviceInput);
    }

    int getBatchSize() const noexcept override
    {
        return mBatchSize;
    }


    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        std::cout << "Getting batches (current: " << mCurrentBatch << ", total: " << mCalibrationImages.size() << ")" << std::endl;

        // Check if we have enough images for at least one batch
        if (mCurrentBatch >= mCalibrationImages.size())
        {
            return false; // No more batches
        }

        // Find the input tensor binding by name (usually "images" for YOLO)
        int inputBindingIndex = -1;
        for (int i = 0; i < nbBindings; ++i)
        {
            if (names[i] != nullptr && std::string(names[i]) == "images")
            {
                inputBindingIndex = i;
                break;
            }
        }
        
        // Fallback to first binding if name not found
        if (inputBindingIndex == -1 && nbBindings > 0)
        {
            inputBindingIndex = 0;
        }
        
        if (inputBindingIndex == -1)
        {
            std::cerr << "No input binding found!" << std::endl;
            return false;
        }

        // Copy current batch to GPU
        // For dynamic shapes with INT8, always provide full batch size (padding with zeros if needed)
        size_t actualBatchSize = std::min(mBatchSize, static_cast<int>(mCalibrationImages.size() - mCurrentBatch));
        size_t imageSize = mInputChannels * mInputHeight * mInputWidth * sizeof(float);
        
        // Clear the entire buffer first (important for partial batches - zeros will pad)
        cudaMemset(mDeviceInput, 0, mBatchSize * imageSize);
        
        // Copy images to GPU
        for (int i = 0; i < actualBatchSize; ++i)
        {
            const auto& img = mCalibrationImages[mCurrentBatch + i];
            
            if (img.size() != static_cast<size_t>(mInputChannels * mInputHeight * mInputWidth))
            {
                std::cerr << "Image size mismatch! Expected " 
                        << (mInputChannels * mInputHeight * mInputWidth) 
                        << " got " << img.size() << std::endl;
                return false;
            }
            
            cudaError_t err = cudaMemcpy(
                static_cast<char*>(mDeviceInput) + i * imageSize,
                img.data(),
                imageSize,
                cudaMemcpyHostToDevice
            );
            
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
        }
        
        // Set the binding for the input tensor
        bindings[inputBindingIndex] = mDeviceInput;
        
        // Ensure CUDA operations complete before returning
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess)
        {
            std::cerr << "CUDA sync failed: " << cudaGetErrorString(syncErr) << std::endl;
            return false;
        }
        
        mCurrentBatch += actualBatchSize;
        std::cout << "Finished getting batch of size " << actualBatchSize << " (requested " << mBatchSize << ")" << std::endl;

        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        std::cout << "Trying to read calibrator cache" << std::endl; 
        if (mCacheFile.empty()) return nullptr;
        
        std::ifstream file(mCacheFile, std::ios::binary);
        if (!file.good()) return nullptr;
        
        file.seekg(0, file.end);
        length = file.tellg();
        file.seekg(0, file.beg);
        
        mCalibrationCache.resize(length);
        file.read(reinterpret_cast<char*>(mCalibrationCache.data()), length);
        return mCalibrationCache.data();
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        std::cout << "Trying to write calibrator cache" << std::endl; 

        if (mCacheFile.empty()) return;
        
        std::ofstream file(mCacheFile, std::ios::binary);
        file.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    void loadCalibrationImages(const std::string& path)
    {
        for (auto const& dir_entry : std::filesystem::directory_iterator{path})
        {
            if (std::filesystem::is_directory(dir_entry)) continue;
            
            // Load image as BGR (OpenCV default)
            cv::Mat image = cv::imread(dir_entry.path().string(), cv::IMREAD_COLOR);
            if (image.empty()) continue;

            // Convert BGR to RGB
            cv::Mat image_rgb;
            cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

            // Resize FIRST on uint8 (matches CImg behavior - resize before normalization)
            // Use INTER_LINEAR to match CImg's default resize interpolation
            cv::Mat resized;
            cv::resize(image_rgb, resized, cv::Size(mInputWidth, mInputHeight), 0, 0, cv::INTER_LINEAR);

            // Normalize to [0, 1] float range AFTER resize (matches CImg normalize(0.0F, 1.0F))
            cv::Mat normalized;
            resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

            // Convert to contiguous float array (HWC layout: height * width * channels)
            std::vector<float> imgData(mInputChannels * mInputHeight * mInputWidth);
            size_t channelSize = mInputHeight * mInputWidth;
            const float* srcData = normalized.ptr<float>();
            
            // Reorganize from [R0,G0,B0, R1,G1,B1, ...] to [R0,R1,..., G0,G1,..., B0,B1,...]
            for (int c = 0; c < mInputChannels; ++c) {
                for (int h = 0; h < mInputHeight; ++h) {
                    for (int w = 0; w < mInputWidth; ++w) {
                        // HWC: srcData[h * width * channels + w * channels + c]
                        // CHW: imgData[c * height * width + h * width + w]
                        imgData[c * channelSize + h * mInputWidth + w] = 
                            srcData[h * mInputWidth * mInputChannels + w * mInputChannels + c];
                    }
                }
            }
            
            mCalibrationImages.push_back(imgData);
        }
    }

    int mBatchSize;
    int mInputHeight, mInputWidth, mInputChannels;
    std::string mCacheFile;
    std::vector<std::vector<float>> mCalibrationImages;
    void* mDeviceInput{nullptr};
    int mCurrentBatch{0};
    std::vector<uint8_t> mCalibrationCache;
};