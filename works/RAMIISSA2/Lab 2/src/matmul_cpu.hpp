#pragma once
#include <vector>

void matmul_cpu(const float* A, const float* B, float* C,
    int M, int N, int K);
