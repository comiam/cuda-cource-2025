NVCC = nvcc
CFLAGS = -O3 -arch=sm_75

TARGET = matmul
SRC_DIR = src

SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/kernel.cu $(SRC_DIR)/utils.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean
