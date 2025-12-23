#pragma once
#include <vector>
#include <cstdint>

template <typename T>
void radix_sort_gpu(T* device_input, T* device_temp, int num_elements);

template <typename T>
void radix_sort_cpu(std::vector<T>& data);
