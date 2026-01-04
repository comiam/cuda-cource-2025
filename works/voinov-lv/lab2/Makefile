NVCC = nvcc
CUDA_ARCH = -arch=sm_70 
CXXFLAGS = -O2 -std=c++14 $(CUDA_ARCH)

SOURCES = src/main.cu
TARGET = main

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean