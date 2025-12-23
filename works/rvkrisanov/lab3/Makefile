NVCC = nvcc
CUDA_ARCH = -arch=sm_70 
CXXFLAGS = -O2 -std=c++14 $(CUDA_ARCH)
INCLUDES = -Iinclude

SOURCES = src/main.cu src/sobel_cuda.cu
TARGET = sobel_filter

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o output_sobel.png

.PHONY: all clean
