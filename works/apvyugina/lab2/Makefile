EIGEN_INCLUDE_DIR=/home/anastasia/DEV/cpp_libs/eigen-5.0.0/
CXXFLAGS= -Wall -std=c++17 -O2

build_cpu:
	g++ $(CXXFLAGS) src/main.cpp -o build/cpu_multplication -I include/ -I $(EIGEN_INCLUDE_DIR)

build_cuda:
	nvcc src/gpu_multiplication.cu -o build/gpu_multiplication

build:
	nvcc -c src/gpu_multiplication.cu -o build/gpu_multiplication.o
	g++ $(CXXFLAGS) -c src/main.cpp -o build/main.o
	cd build
	g++ $(CXXFLAGS) -o multiplication main.o gpu_multiplication.o

clean:
	rm -rf build/* 

.PHONY: clean build_cpu build_cuda build