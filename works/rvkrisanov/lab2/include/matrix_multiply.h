#pragma once
#include <cuda_runtime.h>

void matrix_multiply_cpu(
    const float* matrix_a,
    const float* matrix_b,
    float* matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns);

void launch_matrix_multiply_basic(
    const float* d_matrix_a,
    const float* d_matrix_b,
    float* d_matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns,
    cudaStream_t stream = 0);

void launch_matrix_multiply_tiled(
    const float* d_matrix_a,
    const float* d_matrix_b,
    float* d_matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns,
    cudaStream_t stream = 0);
