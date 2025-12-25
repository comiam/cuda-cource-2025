NVCC = nvcc
CFLAGS = -O2
TARGET = radixsort
SRC = src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean

