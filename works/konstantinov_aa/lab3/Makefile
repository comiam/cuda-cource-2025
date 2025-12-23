NVCC ?= nvcc


CUDA_ARCH ?= -arch=sm_70


OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)


CXXFLAGS ?= -O2 -std=c++14 $(CUDA_ARCH) $(OPENCV_CFLAGS)

SOURCES = src/main_opencv.cpp src/sobel.cu
TARGET = sobel_filter

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(OPENCV_LIBS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

