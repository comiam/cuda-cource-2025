NVCC = nvcc
CFLAGS = -O2 -arch=sm_70 -std=c++14

TARGET = radix_sort
SOURCE = src/radix.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean