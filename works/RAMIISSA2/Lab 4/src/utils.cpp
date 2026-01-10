#include "utils.h"
#include <random>
#include <limits>

std::vector<int32_t> generate_random_ints(size_t n) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max()
    );

    std::vector<int32_t> data(n);
    for (auto& x : data) {
        x = dist(rng);
    }
    return data;
}
