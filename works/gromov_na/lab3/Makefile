NVCC = nvcc
CFLAGS = -O2 -arch=sm_70

TARGET = sobel
SOURCE = src/sobel.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean