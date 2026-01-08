NVCC = nvcc
CXXFLAGS = -O2 -std=c++14 -arch=sm_70
INCLUDES = -Iinclude
STB_NOT_USED = -diag-suppress 550

SOURCES = src/main.cu
TARGET = main

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(STB_NOT_USED)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean