NVCC = nvcc
CFLAGS = -O3 -arch=sm_75 -use_fast_math -std=c++14

TARGET = lab_2

SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC) error_check.cuh
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

