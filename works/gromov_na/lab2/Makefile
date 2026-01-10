NVCC = nvcc
CFLAGS = -O2 -arch=sm_70

TARGET = matrix
SOURCE = src/matrix.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean