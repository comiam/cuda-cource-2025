#include "DetectionModelTRT.h"
#include "Timers.h"

void Logger::log(Severity severity, const char* msg) noexcept
{
    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}


bool DetectionModelTRT::build(){

    std::ifstream file(mParams.engineFileName, std::ios::binary);
    if (file.good()){
        std::cout << "Engine file with such name `" << mParams.engineFileName << "` already exists, exiting." << std::endl;
        return true;
    }
    
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(this->logger));
    assert(builder != nullptr);
    
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    assert(network != nullptr);

    
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    assert(config != nullptr);

    if (mParams.fp16) config->setFlag(BuilderFlag::kFP16);
    if (mParams.bf16) config->setFlag(BuilderFlag::kBF16);
    if (mParams.int8) config->setFlag(BuilderFlag::kINT8);

    enableDLA(builder.get(), config.get(), mParams.dlaCore);

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, this->logger));
    assert(parser != nullptr);
    
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    assert(parsed);

    const auto input = network->getInput(0);
    const auto inputName = input->getName();

    // Specify the optimization profile
    nvinfer1::IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(8, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(16, mParams.inputNChannels, mParams.inputHeight, mParams.inputWidth));
    config->addOptimizationProfile(optProfile);
    
    if (mParams.int8) {
        // Use calibrator if calibration data path is provided
        if (!mParams.calibrationDataPath.empty()) {
            std::cout << "Creating calibrator" << std::endl;
            mCalibrator = std::make_unique<Int8EntropyCalibrator2>(
                mParams.calibrationDataPath,
                1, // batch size for calibration (usually 1)
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

    std::unique_ptr<IHostMemory> plan {builder->buildSerializedNetwork(*network, *config)};
    assert(plan != nullptr);
    

    auto runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(this->logger), InferDeleter());
    assert(runtime != nullptr);


    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    assert(engine != nullptr);


    // save engine to binary file
    std::ofstream outfile(mParams.engineFileName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    std::cout << "Success, saved engine to " << mParams.engineFileName << std::endl;
    cudaStreamDestroy(profileStream);

    return true;
}


bool DetectionModelTRT::load(){
    std::vector<char> trtModelStream_;
    size_t size{0};
    
    std::ifstream file(mParams.engineFileName, std::ios::binary);
    assert(file.good());
    
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
        
    trtModelStream_.resize(size);
    file.read(trtModelStream_.data(), size);
    file.close();
    
    
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->logger));
    assert(mRuntime);
    
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(trtModelStream_.data(), size));
    assert(mEngine);

    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    assert(mContext);
    
    return true;
}



void DetectionModelTRT::detect(
    std::vector<cimg_library::CImg<float>> imgList, 
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

    
    // copy img from CImg instance to HostBuffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i=0; i<imgList.size(); ++i){
        auto img = imgList[i];
        std::copy(img.data(), img.data() + img.size(), hostDataBuffer + i*img.size());
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = mContext->enqueueV3(stream);
    assert(status);

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);
    cudaStreamSynchronize(stream);
    
    rawOutput = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
}

DetectionModelTRT::~DetectionModelTRT(){
    this->exit();
}

void DetectionModelTRT::exit(){
    // очистка буферов выполняется автоматически по определению класса Buffers.h
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


int main(int argc, char** argv)
{
    char* onnxFileName = argv[1];

    std::filesystem::path onnxFilePath(onnxFileName);
    std::string engineFileName = onnxFilePath.replace_extension("engine").string();
  
    Params params;
    
    params.onnxFileName = onnxFileName;
    params.engineFileName = engineFileName.c_str();
    
    params.inputTensorNames.push_back("images");
    params.outputTensorNames.push_back("output0");   

    params.dlaCore = -1; // not supported on the server
    params.int8 = true;
    params.fp16 = false;
    params.bf16 = false;

    params.inputHeight = 640;
    params.inputWidth = 640;
    params.inputNChannels = 3;

    params.outputLength = 300;
    params.outputItemSize = 6;

    params.calibrationDataPath = "assets/";  // Use same images or dedicated calibration set
    params.calibrationCacheFile = "calibration.cache";
    
    DetectionModelTRT Engine(params);

    std::cout << "Building and running a GPU inference engine for " << onnxFileName << std::endl;
    auto status = Engine.build();
    std::cout << std::boolalpha << "Build Engine with status " << status << std::endl;
    
    status = Engine.load();
    std::cout << std::boolalpha << "Load Engine with status " << status << std::endl;
    
    const std::filesystem::path img_path{"assets/"};
    std::vector<cimg_library::CImg<float>> fullImgList = Utility::processInput(params, img_path);
    int numberOfImages = fullImgList.size();
    std::cout << "Total number of Images: " << numberOfImages << std::endl;
    std::cout << std::endl;

    std::mt19937 randomRange(std::random_device{}());
    Timer timer;


    int batchSize = 8;
    std::vector<cimg_library::CImg<float>> randomBatch(batchSize);
    float* rawOutput = nullptr;

    std::vector<double> timePerBatch;
    std::sample(fullImgList.begin(), fullImgList.end(), randomBatch.begin(), batchSize, randomRange);

    timer.tic();
    Engine.detect(randomBatch, rawOutput);
    double diff = timer.toc();
    std::cout << "Batch size=" << batchSize << " took " << diff  << " ms, "  <<
        diff/batchSize << " ms/img" << std::endl;

    std::vector<std::vector<Detection>> resultList = Utility::processOutput(rawOutput, batchSize, params);
        
    assert(batchSize == resultList.size());
    for(int i = 0; i < batchSize; ++i){
        auto img = randomBatch[i];
        auto result = resultList[i];

        std::string filename = "results/trt/" + std::to_string(i) + ".png";
        Utility::drawResult(img, result, filename.c_str());
    }
    
    return 0;
}