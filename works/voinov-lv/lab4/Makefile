NVCC = nvcc
CUDA_ARCH = -arch=sm_86 
CXXFLAGS = -O2 -std=c++17 $(CUDA_ARCH)

SOURCES = src/main.cu
TARGET = main

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean