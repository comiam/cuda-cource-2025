#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <filesystem>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "lab5_types.h"
#include "postprocess_gpu.h"

// Макрос для проверки ошибок CUDA (версия с исключением для C++ кода)
#define CUDA_CHECK(expr)                                                     \
    do {                                                                    \
        cudaError_t err = (expr);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            throw std::runtime_error(cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

using namespace nvinfer1;
namespace fs = std::filesystem;

// Вспомогательная функция для получения размера элемента по типу данных TensorRT
static size_t getElementSize(DataType dtype) {
    switch (dtype) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
        case DataType::kUINT8: return 1;
        case DataType::kFP8:   return 1;
        case DataType::kBF16:  return 2;
        case DataType::kINT64: return 8;
        case DataType::kINT4:  return 1; // упрощённо
        default: return 4; // fallback to float
    }
}

static const char* dataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::kFLOAT: return "FP32";
        case DataType::kHALF:  return "FP16";
        case DataType::kINT8:  return "INT8";
        case DataType::kINT32: return "INT32";
        case DataType::kBOOL:  return "BOOL";
        case DataType::kUINT8: return "UINT8";
        case DataType::kFP8:   return "FP8";
        case DataType::kBF16:  return "BF16";
        case DataType::kINT64: return "INT64";
        case DataType::kINT4:  return "INT4";
        default: return "UNKNOWN";
    }
}

// Logger for TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;





//data cleaning and loading labels
static inline std::string trim_copy(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

static std::vector<std::string> load_labels_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return {};
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        line = trim_copy(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        out.push_back(line);
    }
    return out;
}

static std::string format_label(int classId, int labelOffset, float conf, const std::vector<std::string>& classNames) {
    int idx = classId + labelOffset;
    std::string name;
    if (idx >= 0 && idx < (int)classNames.size()) {
        name = classNames[idx];
    } else {
        name = std::to_string(classId);
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), conf);
    return std::string(buf);
}








static void draw_detections(cv::Mat& img, const std::vector<Detection>& dets, const std::vector<std::string>& classNames, int labelOffset) {
    // Палитра (BGR). Первый цвет — синий (по запросу), остальные дают разный цвет по классу.
    const std::vector<cv::Scalar> palette = {
        cv::Scalar(255, 0, 0),     // blue
        cv::Scalar(255, 128, 0),   // light blue
        cv::Scalar(255, 0, 128),   // purple-ish
        cv::Scalar(255, 255, 0),   // cyan
        cv::Scalar(0, 128, 255),   // orange
        cv::Scalar(0, 255, 255),   // yellow
        cv::Scalar(0, 255, 0),     // green
        cv::Scalar(0, 0, 255),     // red
        cv::Scalar(128, 0, 255),   // magenta
        cv::Scalar(255, 0, 255),   // pink
    };

    for (const auto& d : dets) {
        int x1 = (int)std::round(d.box[0]);
        int y1 = (int)std::round(d.box[1]);
        int x2 = (int)std::round(d.box[2]);
        int y2 = (int)std::round(d.box[3]);

        x1 = std::max(0, std::min(x1, img.cols - 1));
        y1 = std::max(0, std::min(y1, img.rows - 1));
        x2 = std::max(0, std::min(x2, img.cols - 1));
        y2 = std::max(0, std::min(y2, img.rows - 1));

            // Детерминированный выбор цвета по отображаемому индексу (с учетом labelOffset)
        int cid = d.classId + labelOffset;
        if (cid < 0) cid = 0;
        int idx = cid % (int)palette.size();
        cv::Scalar color = palette[idx];

        // Толщина линий (жирнее)
        const int thickness = 3;
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);

        std::string label = format_label(d.classId, labelOffset, d.conf, classNames);
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        int tx = x1;
        int ty = std::max(0, y1 - textSize.height - baseLine - 3);

        cv::Rect bgRect(tx, ty, textSize.width + 6, textSize.height + baseLine + 6);
        bgRect.width = std::min(bgRect.width, img.cols - bgRect.x);
        bgRect.height = std::min(bgRect.height, img.rows - bgRect.y);
        cv::rectangle(img, bgRect, color, cv::FILLED);
        cv::putText(img, label, cv::Point(tx + 3, ty + textSize.height + 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}




class RetinaNetDetector {
public:
    //constructor for RetinaNetDetector
    RetinaNetDetector(const std::string& enginePath) : stream(nullptr) {
        initLibNvInferPlugins(&gLogger, "");
        loadEngine(enginePath);
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    //destructor for RetinaNetDetector
    ~RetinaNetDetector() {
        // Destroy CUDA stream
        if (stream) cudaStreamDestroy(stream);
        // Free all buffers
        for (void* buf : buffers) {
            if (buf) cudaFree(buf);
        }
    }

    std::vector<Detection> detect(const cv::Mat& img, float confThreshold = 0.5f) {
        // 1. Preprocess
        float* hostInputBuffer = nullptr;
        preprocess(img, hostInputBuffer);

        // 2. Inference
        // Set tensor addresses for TRT 10+
        // We set them once or every time? context->setTensorAddress needs to be called.
        // For simple execution, we can loop and set them.
        // We already did this in loadEngine but let's ensure.
        // context->setTensorAddress is persistent.
        
        // Copy input to device with proper type conversion (асинхронно в stream)
        if (inputDtype == DataType::kFLOAT) {
            // FP32: прямое копирование
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], hostInputBuffer, inputSize, cudaMemcpyHostToDevice, stream));
        } else if (inputDtype == DataType::kHALF) {
            // FP16: конвертируем на CPU, затем копируем
            std::vector<__half> fp16Buffer(inputNumElements);
            for (size_t i = 0; i < inputNumElements; ++i) {
                fp16Buffer[i] = __float2half(hostInputBuffer[i]);
            }
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], fp16Buffer.data(), inputSize, cudaMemcpyHostToDevice, stream));
            cudaStreamSynchronize(stream); // Ждём завершения, т.к. fp16Buffer локальный
        } else if (inputDtype == DataType::kINT8) {
            // INT8: простая квантизация (scale=1/127, zero_point=0)
            // В реальности нужно использовать calibration scale из движка
            std::vector<int8_t> int8Buffer(inputNumElements);
            for (size_t i = 0; i < inputNumElements; ++i) {
                float val = hostInputBuffer[i] * 127.0f;
                val = std::max(-127.0f, std::min(127.0f, val));
                int8Buffer[i] = static_cast<int8_t>(val);
            }
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], int8Buffer.data(), inputSize, cudaMemcpyHostToDevice, stream));
            cudaStreamSynchronize(stream); // Ждём завершения, т.к. int8Buffer локальный
            std::cerr << "[WARN] INT8 input: using naive quantization (scale=127). "
                      << "For best results, use engine with FP32 input." << std::endl;
        } else {
            // Fallback: копируем как FP32 (может не работать)
            std::cerr << "[WARN] Unsupported input dtype, copying as FP32" << std::endl;
            CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], hostInputBuffer, inputNumElements * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        
        context->enqueueV3(stream);
        
        // 3. Postprocess
        std::vector<Detection> detections;

        // If engine provides NMS plugin outputs, numDetsIndex will be set.
        // Otherwise, we expect raw outputs: cls_logits + bbox_regression.
        if (numDetsIndex >= 0) {
            // Синхронизация перед чтением результатов
            cudaStreamSynchronize(stream);
            
            int numDetections = 0;
            CUDA_CHECK(cudaMemcpyAsync(&numDetections, buffers[numDetsIndex], sizeof(int), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream); // Нужно знать numDetections для выделения векторов

            std::vector<float> boxes(numDetections * 4);
            std::vector<float> scores(numDetections);
            std::vector<float> classes(numDetections);

            CUDA_CHECK(cudaMemcpyAsync(boxes.data(), buffers[boxesIndex], numDetections * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(scores.data(), buffers[scoresIndex], numDetections * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(classes.data(), buffers[classesIndex], numDetections * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream); // Ждём завершения всех копирований

            for (int i = 0; i < numDetections; ++i) {
                if (scores[i] >= confThreshold) {
                    Detection d;
                    d.box[0] = boxes[i * 4 + 0];
                    d.box[1] = boxes[i * 4 + 1];
                    d.box[2] = boxes[i * 4 + 2];
                    d.box[3] = boxes[i * 4 + 3];
                    d.conf = scores[i];
                    d.classId = static_cast<int>(classes[i]);
                    detections.push_back(d);
                }
            }
        } else {
            // RAW: boxesIndex == cls_logits, scoresIndex == bbox_regression? We'll set explicitly by names later.
            // Here we assume boxesIndex->cls_logits, scoresIndex->bbox_regression, classesIndex unused.
            // RAW: cls_logits + bbox_regression
            // Derive numAnchors/numClasses from output tensor shapes (do NOT hardcode 640x640).
            if (clsTensorName.empty() || bboxTensorName.empty()) {
                throw std::runtime_error("RAW outputs not found (expected tensors: cls_logits and bbox_regression)");
            }

            auto clsDims = engine->getTensorShape(clsTensorName.c_str());
            auto boxDims = engine->getTensorShape(bboxTensorName.c_str());

            auto parseAnchorsClasses = [](const nvinfer1::Dims& d, int& outAnchors, int& outClasses) {
                if (d.nbDims == 3) {          // [N, A, C]
                    outAnchors = d.d[1];
                    outClasses = d.d[2];
                } else if (d.nbDims == 2) {   // [A, C]
                    outAnchors = d.d[0];
                    outClasses = d.d[1];
                } else {
                    outAnchors = -1;
                    outClasses = -1;
                }
            };

            int numAnchors = -1;
            int numClasses = -1;
            parseAnchorsClasses(clsDims, numAnchors, numClasses);
            if (numAnchors <= 0 || numClasses <= 0) {
                throw std::runtime_error("Unexpected cls_logits dims (expected [1,A,C] or [A,C])");
            }

            // Validate bbox dims (expect last dimension == 4)
            int bboxLast = (boxDims.nbDims > 0) ? boxDims.d[boxDims.nbDims - 1] : -1;
            if (bboxLast != 4) {
                std::cerr << "[WARN] Unexpected bbox_regression last dim: " << bboxLast << " (expected 4)\n";
            }
            // GPU постпроцесс: decode + фильтр + сортировка + NMS (topK)
            // maxCandidates/topK можно подкрутить под скорость/качество.
            const int maxCandidates = 6000;
            const int topK = 1500;
            detections = retinanet_postprocess_gpu(
                (const float*)buffers[boxesIndex],
                (const float*)buffers[scoresIndex],
                numAnchors, numClasses,
                inputW, inputH,
                img.cols, img.rows,
                confThreshold,
                0.5f,
                maxCandidates,
                topK
            );
            if (detections.empty()) {
                throw std::runtime_error("GPU postprocess returned 0 detections (GPU-only build: no CPU fallback).");
            }
        }
        
        delete[] hostInputBuffer;
        return detections;
    }

    cv::Size getInputSize() const { return cv::Size(inputW, inputH); }
    int getRawNumClasses() const { return rawNumClasses; }

private:
    std::shared_ptr<ICudaEngine> engine;
    std::shared_ptr<IExecutionContext> context;
    std::vector<void*> buffers;
    cudaStream_t stream;  // Переиспользуемый CUDA stream для инференса
    
    int inputIndex, numDetsIndex, boxesIndex, scoresIndex, classesIndex;
    size_t inputSize;          // размер входного буфера в байтах (с учётом типа данных движка)
    size_t inputNumElements;   // кол-во элементов входа (C*H*W)
    int inputH, inputW, inputC;
    DataType inputDtype = DataType::kFLOAT;
    std::string clsTensorName;
    std::string bboxTensorName;
    int rawNumClasses = -1;

    void loadEngine(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Error reading engine file: " + path);
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);
        file.close();

        std::unique_ptr<IRuntime, void(*)(IRuntime*)> runtime{createInferRuntime(gLogger), [](IRuntime* r){ delete r; }};
        // TRT 10: deserializeCudaEngine returns ICudaEngine* which we delete manually
        engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream.data(), size), [](ICudaEngine* e){ delete e; });
        if (!engine) throw std::runtime_error("Failed to deserialize engine");

        // TRT 10: createExecutionContext returns IExecutionContext*
        context = std::shared_ptr<IExecutionContext>(engine->createExecutionContext(), [](IExecutionContext* c){ delete c; });

        // Setup buffers using Tensor I/O names (TRT 10 style)
        int numTensors = engine->getNbIOTensors();
        buffers.resize(numTensors);
        
        inputIndex = -1;
        numDetsIndex = -1; boxesIndex = -1; scoresIndex = -1; classesIndex = -1;

        for (int i = 0; i < numTensors; ++i) {
            const char* name = engine->getIOTensorName(i);
            TensorIOMode mode = engine->getTensorIOMode(name);
            
            if (mode == TensorIOMode::kINPUT) {
                inputIndex = i;
                auto dims = engine->getTensorShape(name);
                // Assume NCHW
                inputH = dims.d[2];
                inputW = dims.d[3];
                inputC = dims.d[1];
                inputNumElements = (size_t)inputC * inputH * inputW;
                
                // Учитываем реальный тип данных тензора (FP32/FP16/INT8)
                inputDtype = engine->getTensorDataType(name);
                size_t elemSize = getElementSize(inputDtype);
                inputSize = inputNumElements * elemSize;
                
                std::cout << "[TRT] Input tensor '" << name << "' dtype: " << dataTypeToString(inputDtype) 
                          << ", shape: [1," << inputC << "," << inputH << "," << inputW << "]" << std::endl;
                
                cudaMalloc(&buffers[i], inputSize);
                context->setTensorAddress(name, buffers[i]);
            } else {
                // Output mapping
                std::string sName(name);
                if (sName.find("num_detections") != std::string::npos) numDetsIndex = i;
                else if (sName.find("nmsed_boxes") != std::string::npos) boxesIndex = i;
                else if (sName.find("nmsed_scores") != std::string::npos) scoresIndex = i;
                else if (sName.find("nmsed_classes") != std::string::npos) classesIndex = i;
                else if (sName == "cls_logits") {                         // raw head output
                    boxesIndex = i;
                    clsTensorName = name;
                    // Try to parse raw numClasses from static tensor shape.
                    auto d = engine->getTensorShape(name);
                    if (d.nbDims == 3) rawNumClasses = d.d[2]; // [N,A,C]
                    else if (d.nbDims == 2) rawNumClasses = d.d[1]; // [A,C]
                }
                else if (sName == "bbox_regression") {                    // raw head output
                    scoresIndex = i;
                    bboxTensorName = name;
                }
                
                // Allocate based on shape and actual data type
                // Для динамических размеров используем максимальные из профиля оптимизации
                auto dims = engine->getTensorShape(name);
                DataType dtype = engine->getTensorDataType(name);
                size_t elemSize = getElementSize(dtype);
                
                size_t vol = 1;
                bool hasDynamic = false;
                for(int d=0; d<dims.nbDims; ++d) {
                    if (dims.d[d] == -1) {
                        hasDynamic = true;
                        break;
                    }
                }
                
                if (hasDynamic) {
                    // Получаем максимальные размеры из профиля оптимизации
                    int profileIdx = context->getOptimizationProfile();
                    auto maxDims = engine->getProfileShape(name, profileIdx, OptProfileSelector::kMAX);
                    for(int d=0; d<maxDims.nbDims; ++d) {
                        vol *= (maxDims.d[d] > 0) ? maxDims.d[d] : 1;
                    }
                } else {
                    for(int d=0; d<dims.nbDims; ++d) {
                        vol *= dims.d[d];
                    }
                }
                
                cudaMalloc(&buffers[i], vol * elemSize);
                context->setTensorAddress(name, buffers[i]);
                
                std::cout << "[TRT] Output tensor '" << name << "' dtype: " << dataTypeToString(dtype) << std::endl;
            }
        }
        
        // Simple fallback if names don't match exactly (e.g. if using a model with different output names)
        // For this lab, we assume the model follows the standard TRT NMS plugin naming conventions.
    }

    void preprocess(const cv::Mat& img, float*& hostBuffer) {
        // Resize
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(inputW, inputH));
        
        // Convert to float and normalize like torchvision detection models:
        // x = (x/255 - mean) / std, in RGB order
        cv::Mat floatImg;
        resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

        // OpenCV loads BGR; torchvision expects RGB
        cv::cvtColor(floatImg, floatImg, cv::COLOR_BGR2RGB);

        const cv::Scalar mean(0.485, 0.456, 0.406);
        const cv::Scalar stdv(0.229, 0.224, 0.225);
        cv::subtract(floatImg, mean, floatImg);
        cv::divide(floatImg, stdv, floatImg);
        
        // HWC -> CHW
        hostBuffer = new float[inputSize / sizeof(float)];
        
        // Extract channels
        std::vector<cv::Mat> channels(3);
        for (int i = 0; i < 3; ++i) {
            channels[i] = cv::Mat(inputH, inputW, CV_32FC1, hostBuffer + i * inputH * inputW);
        }
        cv::split(floatImg, channels);
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <video_or_image_file> [output_path]" << std::endl;
        return 1;
    }

    std::string enginePath = argv[1];
    std::string inputPath = argv[2];
    std::string outputPath = (argc >= 4) ? argv[3] : "";

    try {
        RetinaNetDetector detector(enginePath);
        std::cout << "Engine loaded successfully." << std::endl;

        // ---- Labels loading & validation ----
        // Labels are REQUIRED.
        // Priority:
        // 1) env RETINANET_LABELS
        // 2) labels.txt next to engine file
        std::vector<std::string> labels;
        int labelOffset = 0; // 0: classId==index; -1: classId in [1..N] (background at 0)

        if (const char* env = std::getenv("RETINANET_LABELS")) {
            auto tmp = load_labels_txt(env);
            if (!tmp.empty()) {
                labels = std::move(tmp);
                std::cout << "Loaded labels from RETINANET_LABELS: " << env << std::endl;
            } else {
                throw std::runtime_error(std::string("RETINANET_LABELS is set but file could not be read or is empty: ") + env);
            }
        } else {
            fs::path p(enginePath);
            fs::path candidate = p.has_parent_path() ? (p.parent_path() / "labels.txt") : fs::path("labels.txt");
            if (fs::exists(candidate)) {
                auto tmp = load_labels_txt(candidate.string());
                if (!tmp.empty()) {
                    labels = std::move(tmp);
                    std::cout << "Loaded labels from: " << candidate.string() << std::endl;
                }
            }
        }

        if (labels.empty()) {
            throw std::runtime_error(
                "labels.txt is required. Put labels.txt next to the .engine file "
                "or set RETINANET_LABELS to a valid labels.txt path."
            );
        }

        int nc = detector.getRawNumClasses();
        if (nc > 0) {
            if (nc == (int)labels.size()) {
                labelOffset = 0;
            } else if (nc == (int)labels.size() + 1) {
                labelOffset = -1;
                std::cout << "[INFO] Using labelOffset=-1 (skip background): numClasses=" << nc
                          << ", labels=" << labels.size() << std::endl;
            } else {
                throw std::runtime_error(
                    "labels mismatch: numClasses(" + std::to_string(nc) +
                    ") must equal labels.size(" + std::to_string(labels.size()) +
                    ") or labels.size()+1 (background)."
                );
            }
        } else {
            throw std::runtime_error("Could not read raw numClasses from engine at load time (cls_logits shape unknown).");
        }

        
        cv::Mat image = cv::imread(inputPath);
        if (!image.empty()) {
            if (outputPath.empty()) outputPath = "output.png";
            auto start = std::chrono::high_resolution_clock::now();
            auto dets = detector.detect(image, 0.5f);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();

            draw_detections(image, dets, labels, labelOffset);
            cv::putText(image, "Inference: " + std::to_string(ms) + " ms",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            if (!cv::imwrite(outputPath, image)) {
                std::cerr << "Failed to write output image: " << outputPath << std::endl;
                return 1;
            }
            std::cout << "Done! Result saved to " << outputPath << std::endl;
            return 0;
        }

        // Otherwise treat as video
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error opening input file (not an image/video?): " << inputPath << std::endl;
            return 1;
        }

        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        if (fps <= 0) fps = 30;

        if (outputPath.empty()) outputPath = "output.mp4";
        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));

        cv::Mat frame;
        int frameCount = 0;

        // Metrics variables
        double totalInferenceTime = 0.0;
        long totalDetections = 0;
        std::vector<Detection> lastDetections;

        auto totalStart = std::chrono::high_resolution_clock::now();

        while (cap.read(frame)) {
            auto start = std::chrono::high_resolution_clock::now();

            std::vector<Detection> dets = detector.detect(frame, 0.5f);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();

            // Accumulate metrics
            totalInferenceTime += ms;
            totalDetections += dets.size();
            lastDetections = dets;

            draw_detections(frame, dets, labels, labelOffset);

            cv::putText(frame, "Inference: " + std::to_string(ms) + " ms",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            writer.write(frame);

            if (++frameCount % 10 == 0) {
                std::cout << "Processed frame " << frameCount << std::endl;
            }
        }
        
        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalSeconds = std::chrono::duration<double>(totalEnd - totalStart).count();

        cap.release();
        writer.release();

        // Print Summary as per README requirements
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Video: " << inputPath << " (" << width << "x" << height << ", " << fps << " FPS)" << std::endl;
        std::cout << "Total Processing Time: " << totalSeconds << " sec" << std::endl;
        std::cout << "Detected Objects: " << totalDetections << std::endl;
        std::cout << "Result: " << outputPath << std::endl;
        std::cout << "\nExample Detections (from last frame):" << std::endl;

        for (size_t i = 0; i < std::min((size_t)5, lastDetections.size()); ++i) {
            const auto& d = lastDetections[i];
            std::string className;
            int idx = d.classId + labelOffset;
            if (idx >= 0 && idx < (int)labels.size()) {
                className = labels[idx];
            } else {
                className = std::to_string(d.classId);
            }
            std::cout << "- " << className << " (conf: " << d.conf << ") [" 
                      << (int)d.box[0] << ", " << (int)d.box[1] << ", " 
                      << (int)d.box[2] << ", " << (int)d.box[3] << "]" << std::endl;
        }
        std::cout << "==================================================" << std::endl;

        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
