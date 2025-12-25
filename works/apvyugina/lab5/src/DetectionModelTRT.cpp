#include "DetectionModelTRT.h"
#include "Timers.h"
#include <opencv2/opencv.hpp>
#include <cstring>

using namespace std;

void Logger::log(Severity severity, const char* msg) noexcept
{
    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING)
        cout << msg << endl;
}


bool DetectionModelTRT::build(){

    ifstream file(mParams.engineFileName, ios::binary);
    if (file.good()){
        cout << "Engine file with such name `" << mParams.engineFileName << "` already exists, exiting." << endl;
        return true;
    }
    
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(this->logger));
    assert(builder != nullptr);
    
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    assert(network != nullptr);

    
    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    assert(config != nullptr);

    if (mParams.fp16) config->setFlag(BuilderFlag::kFP16);
    if (mParams.bf16) config->setFlag(BuilderFlag::kBF16);
    if (mParams.int8) config->setFlag(BuilderFlag::kINT8);

    enableDLA(builder.get(), config.get(), mParams.dlaCore);

    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, this->logger));
    assert(parser != nullptr);
    
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    assert(parsed);

    const auto input = network->getInput(0);
    const auto inputName = input->getName();

    // Specify the optimization profile with dynamic batch sizes
    nvinfer1::IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(8, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(16, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    config->addOptimizationProfile(optProfile);
    
    if (mParams.int8) {
        // Use calibrator if calibration data path is provided
        if (!mParams.calibrationDataPath.empty()) {
            cout << "Creating calibrator" << endl;
            // Set calibration profile explicitly for dynamic shapes
            // Calibration uses the OPT profile dimension (batch size 8)
            config->setCalibrationProfile(optProfile);
            mCalibrator = make_unique<Int8EntropyCalibrator2>(
                mParams.calibrationDataPath,
                8, // batch size for calibration - must match OPT profile dimension
                mParams.inputHeight,
                mParams.inputWidth,
                mParams.inputNChannels,
                mParams.calibrationCacheFile
            );
            config->setInt8Calibrator(mCalibrator.get());
            // Keep calibrator alive until build completes
            // Store in class member or use unique_ptr that outlives build
        } else {
            // Fallback to manual dynamic ranges
            setAllDynamicRanges(network.get(), 127.0F, 127.0F);
        }
    }



    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    auto cudaStreamErrorCode = cudaStreamCreate(&profileStream);
    assert(cudaStreamErrorCode == 0);
    
    config->setProfileStream(profileStream);

    unique_ptr<IHostMemory> plan {builder->buildSerializedNetwork(*network, *config)};
    assert(plan != nullptr);
    

    auto runtime = shared_ptr<nvinfer1::IRuntime>(createInferRuntime(this->logger), InferDeleter());
    assert(runtime != nullptr);


    auto engine = shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    assert(engine != nullptr);


    // save engine to binary file
    ofstream outfile(mParams.engineFileName, ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    cout << "Success, saved engine to " << mParams.engineFileName << endl;
    cudaStreamDestroy(profileStream);

    return true;
}


bool DetectionModelTRT::load(){
    vector<char> trtModelStream_;
    size_t size{0};
    
    ifstream file(mParams.engineFileName, ios::binary);
    assert(file.good());
    
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
        
    trtModelStream_.resize(size);
    file.read(trtModelStream_.data(), size);
    file.close();
    
    
    mRuntime = shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->logger));
    assert(mRuntime);
    
    mEngine = shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(trtModelStream_.data(), size));
    assert(mEngine);

    mContext = unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    assert(mContext);
    
    return true;
}



void DetectionModelTRT::detect(
    vector<cv::Mat> imgList, 
    float*& rawOutput
){
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvinfer1::Dims4 inputDims = {imgList.size(), mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth};
    mContext->setInputShape(mParams.inputTensorNames[0].c_str(), inputDims);

    BufferManager buffers(mEngine, imgList.size(), mContext.get());

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        mContext->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }
    assert(mParams.inputTensorNames.size() == 1); // only one model entrance

    
    // copy img from cv::Mat to HostBuffer
    // Convert from HWC (cv::Mat) to NCHW (model expects: batch, channels, height, width)
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    size_t imgSize = mParams.inputNChannels * mParams.inputHeight * mParams.inputWidth;
    size_t channelSize = mParams.inputHeight * mParams.inputWidth;
    
    for (int batchIdx = 0; batchIdx < imgList.size(); ++batchIdx){
        const cv::Mat& img = imgList[batchIdx];
        assert(img.isContinuous());
        assert(img.rows == mParams.inputHeight && img.cols == mParams.inputWidth && img.channels() == mParams.inputNChannels);
        
        const float* srcData = img.ptr<float>();
        float* dstData = hostDataBuffer + batchIdx * imgSize;
        
        // Convert HWC to CHW: reorganize from [R0,G0,B0, R1,G1,B1, ...] to [R0,R1,..., G0,G1,..., B0,B1,...]
        for (int c = 0; c < mParams.inputNChannels; ++c) {
            for (int h = 0; h < mParams.inputHeight; ++h) {
                for (int w = 0; w < mParams.inputWidth; ++w) {
                    // HWC: srcData[h * width * channels + w * channels + c]
                    // CHW: dstData[c * height * width + h * width + w]
                    dstData[c * channelSize + h * mParams.inputWidth + w] = 
                        srcData[h * mParams.inputWidth * mParams.inputNChannels + w * mParams.inputNChannels + c];
                }
            }
        }
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = mContext->enqueueV3(stream);
    assert(status);

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);
    cudaStreamSynchronize(stream);
    
    // Calculate output size and copy data to persistent buffer
    size_t outputSize = mParams.outputLength * mParams.outputItemSize * imgList.size();
    if (rawOutput == nullptr) {
        rawOutput = new float[outputSize];
    }
    void* hostOutputBuffer = buffers.getHostBuffer(mParams.outputTensorNames[0]);
    if (hostOutputBuffer != nullptr) {
        memcpy(rawOutput, hostOutputBuffer, outputSize * sizeof(float));
    }
    
    cudaStreamDestroy(stream);
}

DetectionModelTRT::~DetectionModelTRT(){
    this->exit();
}

void DetectionModelTRT::exit(){
    // очистка буферов выполняется автоматически по определению класса Buffers.h
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


bool DetectionModelTRT::prepareEngine() {
    cout << "Building and running a GPU inference engine for " << mParams.onnxFileName << endl;
    
    bool status = build();
    cout << boolalpha << "Build Engine with status " << status << endl;
    
    if (!status) {
        return false;
    }
    
    status = load();
    cout << boolalpha << "Load Engine with status " << status << endl;
    
    return status;
}
