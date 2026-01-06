CXXFLAGS=-Wall -O2 -std=c++17

.PHONY: build clean

build:
	mkdir -p build/
	nvcc -std=c++17 -I include/ src/radix_sort.cu src/main.cu -o build/radix_sort

clean:
	rm -rf build/