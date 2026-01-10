NVCC = nvcc
CFLAGS = -O3 -arch=sm_70 -std=c++11 -ccbin g++-10
INCLUDES = -I./include
LIBS = 

TARGET = radix_sort
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/radix_sort.cu
HEADERS = $(INC_DIR)/radix_sort.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	@echo "Compiling $(TARGET)..."
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LIBS)
	@echo "Build complete!"

clean:
	rm -f $(TARGET) *.o *.out
	rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

run: $(TARGET)
	./$(TARGET)

benchmark: $(TARGET)
	@echo "Running benchmark with 100K elements..."
	./$(TARGET) 100000
	@echo ""
	@echo "Running benchmark with 1M elements..."
	./$(TARGET) 1000000
	@echo ""
	@echo "Running benchmark with 10M elements..."
	./$(TARGET) 10000000

test: $(TARGET)
	@echo "Running tests..."
	./$(TARGET) 10000
	./$(TARGET) 100000
	./$(TARGET) 1000000

.PHONY: all clean run benchmark test
