NVCC ?= nvcc

CUDA_ARCH ?= -arch=sm_70

CXXFLAGS ?= -O3 -std=c++14 $(CUDA_ARCH)

SOURCES = src/rs_opt.cu
TARGET = radix_sort

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

