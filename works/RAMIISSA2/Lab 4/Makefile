CUDA_PATH ?= /usr/local/cuda

NVCC := "$(CUDA_PATH)/bin/nvcc"
CXX := g++

NVCCFLAGS := -O3 -std=c++17
INCLUDES := -Iinclude

TARGET := radix_sort
SRC := src/main.cu src/benchmark.cu src/cpu_sort.cpp src/utils.cpp src/thrust_sort.cu src/radix_sort.cu

all:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
