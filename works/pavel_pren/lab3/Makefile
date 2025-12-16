CC = nvcc
CFLAGS = -O2
TARGET = sobel

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): src/sobel.cu
	$(CC) $(CFLAGS) -o $(TARGET) src/sobel.cu

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
