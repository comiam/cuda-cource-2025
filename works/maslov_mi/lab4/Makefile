# Makefile для CUDA-проекта (Radix Sort)
# Структура:
# src/   <- исходники (*.cu)
# bin/   <- скомпилированный бинарник

NVCC       := nvcc
CUDACFLAGS := -O3 -std=c++17 

SRC_DIR := src
BIN_DIR := bin

TARGET := $(BIN_DIR)/lab4
SRC    := $(SRC_DIR)/lab4.cu

.PHONY: all clean dirs run

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BIN_DIR)

$(TARGET): $(SRC)
	$(NVCC) $(CUDACFLAGS) -o $@ $<

run: all
	./$(TARGET)

clean:
	-rm -f $(TARGET)
	-rm -rf $(BIN_DIR)
