CXXFLAGS=-Wall -O2 -std=c++17
INCLUDES=-I include/

.PHONY: build_sobel clean

build_sobel:
	mkdir -p build/
	g++ $(CXXFLAGS) -I include/ -c src/image_utils.cpp -o build/image_utils.o
	nvcc -std=c++17 -I include/ -c src/gpu_sobel.cu -o build/gpu_sobel.o
	nvcc -std=c++17 -I include/ -c src/cpu_sobel.cpp -o build/cpu_sobel.o
	nvcc build/image_utils.o build/gpu_sobel.o build/cpu_sobel.o -o build/sobel

clean:
	rm -rf build/