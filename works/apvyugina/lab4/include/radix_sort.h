#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include <stdint.h>

void radixSort_int32(uint32_t* d_input, uint32_t* d_output, int n);
void radixSort_int64(uint64_t* d_input, uint64_t* d_output, int n);

#endif // RADIX_SORT_H

