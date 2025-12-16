NVCC = nvcc
CFLAGS = -O3 -use_fast_math -std=c++14
INCLUDES = -Iheaders -Isrc

# Auto-detect compute capability or use default
# You can override: make COMPUTE_CAP=86
COMPUTE_CAP ?= 75
ARCH_FLAGS = -arch=sm_$(COMPUTE_CAP)

# Detect OS and set executable extension
ifeq ($(OS),Windows_NT)
    TARGET = sobel.exe
    RM = del /Q
else
    TARGET = sobel
    RM = rm -f
endif

SRC_DIR = src
HEADERS_DIR = headers

SRC = $(SRC_DIR)/main.cu $(SRC_DIR)/kernel.cu $(SRC_DIR)/utils.cu $(SRC_DIR)/lodepng.cpp
HEADERS = $(HEADERS_DIR)/utils.h $(HEADERS_DIR)/error_check.cuh $(HEADERS_DIR)/lodepng.h

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(CFLAGS) $(ARCH_FLAGS) $(INCLUDES) -o $(TARGET) $(SRC)

clean:
	$(RM) $(TARGET) 2>nul || true

.PHONY: all clean
