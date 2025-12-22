#include "../include/matrix_multiply.h"

void matrix_multiply_cpu(
    const float* matrix_a,
    const float* matrix_b,
    float* matrix_result,
    const int matrix_a_rows,
    const int matrix_a_columns,
    const int matrix_b_columns)
{
    for (int row = 0; row < matrix_a_rows; ++row)
    {
        for (int column = 0; column < matrix_b_columns; ++column)
        {
            float sum = 0.0f;
            for (int k = 0; k < matrix_a_columns; ++k)
            {
                sum += matrix_a[row * matrix_a_columns + k] * matrix_b[k * matrix_b_columns + column];
            }
            matrix_result[row * matrix_b_columns + column] = sum;
        }
    }
}

