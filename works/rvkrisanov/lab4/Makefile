CXX := g++
NVCC := nvcc
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS := -O3 -std=c++17 --gpu-architecture=compute_75

NVCC_PATH := $(shell which nvcc)

# Directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
BIN_DIR := .

# Targets
TARGET := radix_sort

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

benchmark: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

cpu:
	@echo "Lab 4 requires CUDA (nvcc) for Thrust and Radix Sort kernels."
