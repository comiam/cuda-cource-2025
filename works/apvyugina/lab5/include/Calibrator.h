#include "NvInfer.h"
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>  // Add this
#include <iostream>    // Add this

#define cimg_use_png   // Add this if not already included elsewhere
#include "CImg.h"      // Add this if not already included

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
        std::cout << "Getting batches" << std::endl;

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
        size_t actualBatchSize = std::min(mBatchSize, static_cast<int>(mCalibrationImages.size() - mCurrentBatch));
        size_t imageSize = mInputChannels * mInputHeight * mInputWidth * sizeof(float);
        
        // Clear the entire buffer first (important for partial batches)
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
        std::cout << "finished getting batches" << std::endl;

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
            cimg_library::CImg<float> image((char*)dir_entry.path().c_str());
            if (image.is_empty()) continue;

            image.normalize(0.0F, 1.0F);
            image.resize(mInputWidth, mInputHeight);
            
            // Convert to contiguous float array
            std::vector<float> imgData(mInputChannels * mInputHeight * mInputWidth);
            std::copy(image.data(), image.data() + image.size(), imgData.data());
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