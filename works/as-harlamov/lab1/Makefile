NVCC = nvcc
CFLAGS = -O2 -std=c++17 -arch=sm_70

TARGET = circle

all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

.PHONY: all clean