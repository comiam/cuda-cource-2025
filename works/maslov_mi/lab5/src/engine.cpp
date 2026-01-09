#include "engine.h"
#include "calibrator.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

InferenceEngine::InferenceEngine(const ModelOptions& options) : m_options(options) {}

InferenceEngine::~InferenceEngine() {}

bool InferenceEngine::build() {
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

    if (!parser->parseFromFile(m_options.onnx_path.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
        std::cerr << "Не удалось распарсить ONNX файл." << std::endl;
        return false;
    }

    auto profile = builder->createOptimizationProfile();
    Dims4 inputDims(m_options.max_batch_size, 3, m_options.input_height, m_options.input_width);
    
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 3, m_options.input_height, m_options.input_width));
    profile->setDimensions("input", OptProfileSelector::kOPT, inputDims);
    profile->setDimensions("input", OptProfileSelector::kMAX, inputDims);
    config->addOptimizationProfile(profile);

    if (m_options.use_int8 && builder->platformHasFastInt8()) {
        config->setFlag(BuilderFlag::kINT8);

        
        m_calibrator = std::make_unique<Int8EntropyCalibrator2>(
            m_options.max_batch_size, 
            m_options.input_width, 
            m_options.input_height, 
            m_options.calibration_data_path,
            "calib.table",
            "input"
        );
        config->setInt8Calibrator(m_calibrator.get());
        
        std::cout << "Включен режим INT8 c калибратором." << std::endl;
    } else if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "Включен режим FP16." << std::endl;
    }

    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) return false;

    std::ofstream outfile(m_options.engine_path, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    
    std::cout << "Engine сохранен в " << m_options.engine_path << std::endl;
    return true;
}

bool InferenceEngine::load() {
    std::ifstream file(m_options.engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) return false;
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> modelStream(size);
    file.read(modelStream.data(), size);

    m_runtime.reset(createInferRuntime(gLogger));
    m_engine.reset(m_runtime->deserializeCudaEngine(modelStream.data(), size));
    m_context.reset(m_engine->createExecutionContext());
    
    return true;
}

bool InferenceEngine::run(void** buffers, cudaStream_t stream) {
    m_context->setInputShape("input", Dims4(m_options.max_batch_size, 3, m_options.input_height, m_options.input_width));
    
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        m_context->setTensorAddress(name, buffers[i]);
    }
    
    return m_context->enqueueV3(stream);
}
