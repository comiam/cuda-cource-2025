#include "cpu_sort.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

// std::sort baseline
void cpu_sort_std(std::vector<int32_t>& data) {
    std::sort(data.begin(), data.end());
}

// qsort comparison function
static int compare_int32(const void* a, const void* b) {
    int32_t x = *static_cast<const int32_t*>(a);
    int32_t y = *static_cast<const int32_t*>(b);
    return (x > y) - (x < y);
}

// qsort baseline
void cpu_sort_qsort(std::vector<int32_t>& data) {
    qsort(
        data.data(),
        data.size(),
        sizeof(int32_t),
        compare_int32
    );
}
