
NVCC = nvcc
CXX = g++
CFLAGS = -O2 -arch=sm_75
INCLUDES = -Iinclude

TARGET = graphics
OBJS = main.o shapes.o

all: $(TARGET)

$(TARGET): $(OBJS)
 $(NVCC) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: src/main.cpp include/shapes.h
 $(CXX) -c -O2 $(INCLUDES) src/main.cpp -o main.o

shapes.o: src/shapes.cu include/shapes.h
 $(NVCC) $(CFLAGS) $(INCLUDES) -c src/shapes.cu -o shapes.o

clean:
 rm -f $(TARGET) $(OBJS)

.PHONY: all clean
