# Makefile for Sobel Edge Detection Lab

NVCC = nvcc
SRC_DIR = src
HEADERS_DIR = headers

# Проверяем наличие stb_image библиотек
STB_IMAGE_EXISTS = $(wildcard $(HEADERS_DIR)/stb_image.h)
STB_IMAGE_WRITE_EXISTS = $(wildcard $(HEADERS_DIR)/stb_image_write.h)

# Если обе библиотеки доступны, добавляем флаг
ifeq ($(and $(STB_IMAGE_EXISTS),$(STB_IMAGE_WRITE_EXISTS)),)
    NVCC_FLAGS = -O3 -arch=sm_75 -std=c++11 -Iheaders
    $(info PNG support disabled: stb_image libraries not found)
    $(info Run: powershell -ExecutionPolicy Bypass -File download_stb.ps1)
else
    NVCC_FLAGS = -O3 -arch=sm_75 -std=c++11 -Iheaders -DSTB_IMAGE_AVAILABLE
    $(info PNG support enabled: stb_image libraries found)
endif

TARGET = lab_3
SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/sobel.cu $(SRC_DIR)/image_io.cpp
OBJECTS = $(SRC_DIR)/main.o $(SRC_DIR)/sobel.o $(SRC_DIR)/image_io.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(OBJECTS)

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(SRC_DIR)/sobel.o: $(SRC_DIR)/sobel.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(SRC_DIR)/image_io.o: $(SRC_DIR)/image_io.cpp
	$(NVCC) $(NVCC_FLAGS) -x c++ -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

# Для Windows (если используется)
clean-win:
	del /Q $(TARGET).exe $(OBJECTS) 2>nul || true

