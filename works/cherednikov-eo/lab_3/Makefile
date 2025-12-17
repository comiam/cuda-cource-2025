CC = nvcc
CFLAGS = -O2
TARGET = sobel

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): src/main.cu src/image_io.cu
	$(CC) $(CFLAGS) -Iinclude -o $(TARGET) src/main.cu src/image_io.cu

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)