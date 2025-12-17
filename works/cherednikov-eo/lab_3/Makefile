CC = nvcc
CFLAGS = -O2
TARGET = main

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): src/main.cu
	$(CC) $(CFLAGS) -o $(TARGET) src/main.cu

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)