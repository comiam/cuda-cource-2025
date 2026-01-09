cmake_minimum_required(VERSION 3.10)
project(RetinaTRT)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(RetinaTRT src/main.cpp src/engine.cpp src/utils.cpp src/calibrator.cpp)

target_include_directories(RetinaTRT PRIVATE include)

target_link_libraries(RetinaTRT PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
    nvonnxparser
)
