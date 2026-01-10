NVCC = nvcc
CFLAGS = -O2 -arch=sm_70

TARGET = circle

all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) $(CFLAGS) src/main.cu -o $(TARGET)

clean:
	rm -f $(TARGET) *.o
