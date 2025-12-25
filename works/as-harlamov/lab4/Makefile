NVCC = nvcc
CFLAGS = -O2 -arch=sm_70 -std=c++11
LDFLAGS = -lcudart

TARGET = radix_sort

SOURCES = src/main.cu src/sort.cu
HEADERS = src/sort.h src/utils.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

