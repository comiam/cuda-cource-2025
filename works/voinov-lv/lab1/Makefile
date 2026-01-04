NVCC = nvcc
CXXFLAGS = -O2 -std=c++14 -arch=sm_70

SOURCES = src/main.cu
TARGET = main

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean