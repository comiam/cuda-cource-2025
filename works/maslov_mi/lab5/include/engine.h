#pragma once
#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include "utils.h"

struct ModelOptions {
    bool use_int8 = false;
    int max_batch_size = 1;
    int input_width = 640;
    int input_height = 640;
    std::string onnx_path;
    std::string engine_path = "model.engine";
    std::string calibration_data_path; // Video file for calibration
};

class Int8EntropyCalibrator2; // Forward decl

class InferenceEngine {
public:
    InferenceEngine(const ModelOptions& options);
    ~InferenceEngine();

    bool build();
    bool load();
    bool run(void** buffers, cudaStream_t stream);

    nvinfer1::Dims3 getInputDims() const;

private:
    ModelOptions m_options;
    
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    
    // Hold calibrator instance to ensure it lives during build
    std::unique_ptr<Int8EntropyCalibrator2> m_calibrator;
};
