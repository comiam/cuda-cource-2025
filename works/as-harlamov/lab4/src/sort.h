#ifndef SORT_H__
#define SORT_H__

#include <cstdint>

// Шаблонная функция для поддержки 32-битных и 64-битных чисел
template<typename T>
void radix_sort(T* const device_output,
    T* const device_input,
    unsigned int input_length);

#endif