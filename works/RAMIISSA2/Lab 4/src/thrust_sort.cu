#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

void warmup_thrust() {
    thrust::device_vector<int> tmp(1);
    thrust::sort(tmp.begin(), tmp.end());
}

void thrust_sort_gpu(
    const std::vector<int32_t>& input,
    std::vector<int32_t>& output,
    float& gpu_time_ms
) {
    size_t n = input.size();
    output.resize(n);

    int32_t* d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(int32_t));

    // Copy input to device
    cudaMemcpy(d_data, input.data(),
        n * sizeof(int32_t),
        cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Sort timing
    cudaEventRecord(start);
    thrust::sort(thrust::device_ptr<int32_t>(d_data),
        thrust::device_ptr<int32_t>(d_data + n));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Copy result back
    cudaMemcpy(output.data(), d_data,
        n * sizeof(int32_t),
        cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
