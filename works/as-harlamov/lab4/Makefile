NVCC = nvcc
CFLAGS = -O2 -arch=sm_70 -std=c++11
LDFLAGS = -lcudart

TARGET = radix_sort

SOURCES = main.cu sort.cu
HEADERS = sort.h utils.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

