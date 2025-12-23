NVCC = nvcc
CFLAGS = -O2 -arch=sm_70

TARGET = square

all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
.PHONY: all clean

