NVCC = nvcc
CFLAGS = -O2 -arch=sm_70 -std=c++14
INCLUDES = -Iinclude
SOURCES = src/main.cu src/matrix_multiply_cpu.cu src/matrix_multiply_cuda.cu src/matrix_multiply_cuda_tiled.cu
TARGET = matrix_multiply

all: $(TARGET) run

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
