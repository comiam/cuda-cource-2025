cmake_minimum_required(VERSION 3.18)
project(retinanet_detection LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(TENSORRT_DIR $ENV{TENSORRT_DIR})
if(NOT TENSORRT_DIR)
    set(TENSORRT_DIR "/usr/local/TensorRT")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_LIBRARY_DIRS}
)

add_executable(retinanet_detection
    src/main.cpp
    src/retinanet.cu
)

target_link_libraries(retinanet_detection
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
)

target_compile_options(retinanet_detection PRIVATE
    -O3
    -Wall
)

