cmake_minimum_required(VERSION 3.18)
project(lab5_retinanet LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)


find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TRT_ROOT} $ENV{TRT_ROOT}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIB nvinfer_10
    HINTS ${TRT_ROOT} $ENV{TRT_ROOT}
    PATH_SUFFIXES lib)

find_library(TENSORRT_PARSERS_LIB nvonnxparser_10
    HINTS ${TRT_ROOT} $ENV{TRT_ROOT}
    PATH_SUFFIXES lib)

find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin_10
    HINTS ${TRT_ROOT} $ENV{TRT_ROOT}
    PATH_SUFFIXES lib)

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(STATUS "TRT_ROOT is set to: ${TRT_ROOT}")
    message(STATUS "Searching for NvInfer.h in: ${TRT_ROOT}/include")
    message(STATUS "Searching for nvinfer.lib in: ${TRT_ROOT}/lib")
    message(FATAL_ERROR "TensorRT not found. Please set TRT_ROOT environment variable.")
endif()

include_directories(${TENSORRT_INCLUDE_DIR} src)

add_executable(retinanet_det src/main.cpp)

target_link_libraries(retinanet_det
    ${OpenCV_LIBS}
    ${TENSORRT_LIB}
    ${TENSORRT_PARSERS_LIB}
    ${TENSORRT_PLUGIN_LIB}
    CUDA::cudart
)

target_sources(retinanet_det PRIVATE src/postprocess_gpu.cu)
set_target_properties(retinanet_det PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 17
)


if(WIN32)
endif()

