cmake_minimum_required(VERSION 3.18)

project(Lab3_Sobel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(SOURCES
    src/sobel.cu
    src/image_io.cpp
)

add_executable(sobel_gpu ${SOURCES})

target_include_directories(sobel_gpu PRIVATE include)

set_target_properties(sobel_gpu PROPERTIES
    CUDA_ARCHITECTURES native
)

set(SOURCES_SHARED
    src/sobel_shared.cu
    src/image_io.cpp
)

add_executable(sobel_gpu_shared ${SOURCES_SHARED})

target_include_directories(sobel_gpu_shared PRIVATE include)

set_target_properties(sobel_gpu_shared PROPERTIES
    CUDA_ARCHITECTURES native
)
