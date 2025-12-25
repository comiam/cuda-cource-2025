#include "../include/matrix_multiply.h"
#include "../include/error_check.cuh"

#define TILE_SIZE 16

__global__ void matrix_multiply_kernel_tiled(
    const float *matrix_a,
    const float *matrix_b,
    float *matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns)
{
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float dot_product = 0.0f;
    const int num_tiles = (matrix_a_columns + TILE_SIZE - 1) / TILE_SIZE;

    #pragma unroll 8
    for (int tile_index = 0; tile_index < num_tiles; ++tile_index)
    {
        const int tiled_a_col = tile_index * TILE_SIZE + tx;
        const int tiled_b_row = tile_index * TILE_SIZE + ty;

        tile_a[ty][tx] = (row < matrix_a_rows && tiled_a_col < matrix_a_columns)
            ? matrix_a[row * matrix_a_columns + tiled_a_col]
            : 0.0f;

        tile_b[ty][tx] = (col < matrix_b_columns && tiled_b_row < matrix_a_columns)
            ? matrix_b[tiled_b_row * matrix_b_columns + col]
            : 0.0f;

        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k)
            dot_product += tile_a[ty][k] * tile_b[k][tx];

        __syncthreads();
    }

    if (row < matrix_a_rows && col < matrix_b_columns)
        matrix_result[row * matrix_b_columns + col] = dot_product;
}

void launch_matrix_multiply_tiled(
    const float *d_matrix_a,
    const float *d_matrix_b,
    float *d_matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns,
    cudaStream_t stream)
{
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (matrix_b_columns + block_size.x - 1) / block_size.x,
        (matrix_a_rows + block_size.y - 1) / block_size.y);

    matrix_multiply_kernel_tiled<<<grid_size, block_size, 0, stream>>>(
        d_matrix_a, d_matrix_b, d_matrix_result,
        matrix_a_rows, matrix_a_columns, matrix_b_columns);

    CUDA_CHECK(cudaGetLastError());
}
