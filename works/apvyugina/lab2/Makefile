EIGEN_INCLUDE_DIR=/home/jovyan/workspace/eigen-5.0.0/
CXXFLAGS= -Wall -std=c++17 -O2

build_cpu:
	mkdir -p build
	g++ $(CXXFLAGS) src/cpu_multiplication.cpp -o build/cpu_multiplication -I include/ -I $(EIGEN_INCLUDE_DIR) -pthread

build_cuda:
	mkdir -p build
	nvcc src/gpu_multiplication.cu -o build/gpu_multiplication -I include/

clean:
	rm -rf build/* 

.PHONY: clean build_cpu build_cuda build