#include "calibrator.h"
#include <fstream>
#include <iterator>

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchSize, int inputW, int inputH, 
                                             const std::string& calibrationDataPath, 
                                             const std::string& calibTableName,
                                             const std::string& inputBlobName, 
                                             bool readCache)
    : mBatchSize(batchSize), mInputW(inputW), mInputH(inputH), 
      mCalibTableName(calibTableName), mInputBlobName(inputBlobName), mReadCache(readCache) {
    
    // Allocate GPU memory for input
    checkCudaErrors(cudaMalloc(&mDeviceInput, mBatchSize * 3 * mInputW * mInputH * sizeof(float)));
    
    // Open video for calibration
    mCap.open(calibrationDataPath);
    if (!mCap.isOpened()) {
        std::cerr << "[WARN] Calibrator could not open video: " << calibrationDataPath << std::endl;
        mTotalFrames = 0;
    } else {
        mTotalFrames = (int)mCap.get(cv::CAP_PROP_FRAME_COUNT);
    }
    
    mCurrentFrame = 0;
    mCurrentBatch = 0;
    mMaxCalibrationBatches = 100; // Calibrate on first 100 batches
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    checkCudaErrors(cudaFree(mDeviceInput));
    if (mCap.isOpened()) mCap.release();
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept {
    return mBatchSize;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (mCurrentBatch >= mMaxCalibrationBatches) return false;
    if (!mCap.isOpened()) return false;

    std::vector<cv::Mat> batchFrames;
    for (int i = 0; i < mBatchSize; ++i) {
        cv::Mat frame;
        if (mCap.read(frame)) {
            batchFrames.push_back(frame);
        } else {
            if (batchFrames.empty()) return false;
            break;
        }
    }
    
    if (batchFrames.empty()) return false;

    // Fill the rest of the batch if video ended in the middle of a batch
    while (batchFrames.size() < mBatchSize) {
        batchFrames.push_back(batchFrames.back().clone());
    }

    
    cv::Mat blob;
    cv::dnn::blobFromImages(batchFrames, blob, 1.0/255.0, cv::Size(mInputW, mInputH), cv::Scalar(0,0,0), true, false);
    
    checkCudaErrors(cudaMemcpy(mDeviceInput, blob.ptr<float>(), 
                              mBatchSize * 3 * mInputW * mInputH * sizeof(float), 
                              cudaMemcpyHostToDevice));

    bindings[0] = mDeviceInput;
    
    std::cout << "[Calibrator] Processing batch " << mCurrentBatch << "/" << mMaxCalibrationBatches << std::endl;
    mCurrentBatch++;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept {
    if (!mReadCache) return nullptr;
    
    mCalibrationCache.clear();
    std::ifstream input(mCalibTableName, std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
    }
    length = mCalibrationCache.size();
    return length ? mCalibrationCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream output(mCalibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

