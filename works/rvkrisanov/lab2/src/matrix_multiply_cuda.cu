#include "../include/matrix_multiply.h"
#include "../include/error_check.cuh"

__global__ void matrix_multiply_kernel_basic(
    const float *matrix_a,
    const float *matrix_b,
    float *matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns)
{
    const int result_row_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int result_column_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (result_row_index >= matrix_a_rows || result_column_index >= matrix_b_columns)
    {
        return;
    }

    float acc_dot_product = 0.0f;

    for (int offset = 0; offset < matrix_a_columns; ++offset)
    {
        const int matrix_a_index = result_row_index * matrix_a_columns + offset;
        const int matrix_b_index = offset * matrix_b_columns + result_column_index;

        acc_dot_product += matrix_a[matrix_a_index] * matrix_b[matrix_b_index];
    }

    const int dot_product_index = result_row_index * matrix_b_columns + result_column_index;
    matrix_result[dot_product_index] = acc_dot_product;
}

void launch_matrix_multiply_basic(
    const float *d_matrix_a,
    const float *d_matrix_b,
    float *d_matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns,
    cudaStream_t stream)
{
    const int tile_size = 16;
    dim3 block_size(tile_size, tile_size);
    dim3 grid_size(
        (matrix_b_columns + block_size.x - 1) / block_size.x,
        (matrix_a_rows + block_size.y - 1) / block_size.y);

    matrix_multiply_kernel_basic<<<grid_size, block_size, 0, stream>>>(
        d_matrix_a, d_matrix_b, d_matrix_result,
        matrix_a_rows, matrix_a_columns, matrix_b_columns);

    CUDA_CHECK(cudaGetLastError());
}
