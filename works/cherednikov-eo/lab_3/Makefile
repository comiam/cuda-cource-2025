NVCC = nvcc
CFLAGS = -O3 -arch=sm_75 -use_fast_math -std=c++14

TARGET = sobel

SRC = src/main.cu src/kernel.cu src/utils.cu src/lodepng.cpp

all: $(TARGET)

$(TARGET): $(SRC) headers/utils.h
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean