CC = nvcc
CFLAGS = -O2
TARGET = radix

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): src/main.cu src/radix_sort.cu
	$(CC) $(CFLAGS) -Iinclude -o $(TARGET) src/main.cu src/radix_sort.cu

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)