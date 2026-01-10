#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (expr);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "cuda error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// cpu умножение: c = a * b
void cpu_matmul(const float *a, const float *b, float *c, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l)
            {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// базовое gpu умножение
__global__ void gpu_matmul_naive(const float *a, const float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        // некоалесцированный доступ к матрице b: потоки в одном warp обращаются к разным строкам
        for (int l = 0; l < k; ++l)
        {
            sum += a[row * k + l] * b[l * n + col];
        }
        c[row * n + col] = sum;
    }
}

// функция транспонирования матрицы на gpu
__global__ void transpose_matrix(const float *input, float *output, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        // записываем input[idy][idx] в output[idx][idy], транспонируя матрицу
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// базовое gpu умножение с коалесцированным доступом к памяти
__global__ void gpu_matmul_naive_coalesced(const float *a, const float *b_transposed, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        // коалесцированный доступ к транспонированной матрице b:
        // потоки в одном warp обращаются к последовательным элементам в памяти
        for (int l = 0; l < k; ++l)
        {
            sum += a[row * k + l] * b_transposed[col * k + l];
        }
        c[row * n + col] = sum;
    }
}

// умножение с коалесцированным доступом к обеим матрицам
// вычисляем (A * B)^T = B^T * A^T, затем транспонируем результат
__global__ void gpu_matmul_coalesced_both(const float *a_transposed, const float *b_transposed, float *c_transposed, int m, int n, int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // используем col для индекса строки в транспонированной матрице
    int row = blockIdx.y * blockDim.y + threadIdx.y; // используем row для индекса столбца в транспонированной матрице

    if (row < n && col < m)
    { // после транспонирования размеры меняются местами
        float sum = 0.0f;
        // при доступе к транспонированным матрицам оба доступа коалесцированные:
        // a_transposed[l * m + col] - коалесцированный доступ к транспонированной A (правильная индексация)
        // b_transposed[row * k + l] - коалесцированный доступ к транспонированной B
        for (int l = 0; l < k; ++l)
        {
            sum += a_transposed[l * m + col] * b_transposed[row * k + l];
        }
        // результат записывается как транспонированный
        c_transposed[col * n + row] = sum;
    }
}

// оптимизированное gpu умножение с shared memory
__global__ void gpu_matmul_shared(const float *a, const float *b, float *c, int m, int n, int k)
{
    const int tile_size = 16;
    // добавляем padding к размеру shared memory
    __shared__ float as[tile_size][tile_size + 1];
    __shared__ float bs[tile_size][tile_size + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    float sum = 0.0f;

    for (int t = 0; t < (k + tile_size - 1) / tile_size; ++t)
    {
        // загрузка в shared memory
        if (row < m && t * tile_size + tx < k)
            as[ty][tx] = a[row * k + t * tile_size + tx];
        else
            as[ty][tx] = 0.0f;

        if (col < n && t * tile_size + ty < k)
            bs[ty][tx] = b[(t * tile_size + ty) * n + col];
        else
            bs[ty][tx] = 0.0f;

        __syncthreads();

        // умножение
        for (int l = 0; l < tile_size; ++l)
        {
            sum += as[ty][l] * bs[l][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[row * n + col] = sum;
    }
}

int main()
{
    int m = 0, n = 0, k = 0;

    std::cout << "enter matrix sizes m, n, k (e.g., 1024 1024 1024): ";
    if (!(std::cin >> m >> n >> k) || m <= 0 || n <= 0 || k <= 0)
    {
        std::cerr << "invalid dimensions.\n";
        return 1;
    }

    std::cout << "matrix size: " << m << "x" << k << " * " << k << "x" << n << "\n";

    const int size_a = m * k;
    const int size_b = k * n;
    const int size_at = k * m; // размер для транспонированной матрицы a
    const int size_bt = n * k; // размер для транспонированной матрицы b
    const int size_ct = n * m; // размер для транспонированной матрицы c (промежуточный результат)
    const int size_c = m * n;

    std::vector<float> h_a(size_a, 1.0f);
    std::vector<float> h_b(size_b, 2.0f);
    std::vector<float> h_c(size_c, 0.0f);

    float *d_a, *d_b, *d_c, *d_at, *d_bt, *d_ct; // указатели на транспонированные и промежуточные матрицы

    // объявление CUDA событий для измерения времени выполнения GPU операций
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // выделение памяти для всех массивов
    CUDA_CHECK(cudaMalloc(&d_a, size_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, size_b * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, size_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_at, size_at * sizeof(float))); // транспонированная матрица a
    CUDA_CHECK(cudaMalloc(&d_bt, size_bt * sizeof(float))); // транспонированная матрицы b
    CUDA_CHECK(cudaMalloc(&d_ct, size_ct * sizeof(float))); // транспонированный результат (промежуточный)

    // копирование исходных данных на gpu
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice));

    // транспонирование матриц a и b на gpu
    dim3 block_transpose(16, 16);
    dim3 grid_transpose_a((k + block_transpose.x - 1) / block_transpose.x,
                          (m + block_transpose.y - 1) / block_transpose.y);
    dim3 grid_transpose_b((n + block_transpose.x - 1) / block_transpose.x,
                          (k + block_transpose.y - 1) / block_transpose.y);

    transpose_matrix<<<grid_transpose_a, block_transpose>>>(d_a, d_at, m, k);
    transpose_matrix<<<grid_transpose_b, block_transpose>>>(d_b, d_bt, k, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    // cpu
    auto start = std::chrono::high_resolution_clock::now();
    cpu_matmul(h_a.data(), h_b.data(), h_c.data(), m, n, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<double>(end - start).count();

    // gpu naive
    CUDA_CHECK(cudaEventRecord(start_event));
    gpu_matmul_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float gpu_naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_naive_time, start_event, stop_event));

    // gpu naive с коалесцированным доступом
    CUDA_CHECK(cudaEventRecord(start_event));
    gpu_matmul_naive_coalesced<<<grid, block>>>(d_a, d_bt, d_c, m, n, k);
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float gpu_coalesced_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_coalesced_time, start_event, stop_event));

    // gpu с коалесцированным доступом к обеим матрицам
    CUDA_CHECK(cudaEventRecord(start_event));
    // вычисляем (A*B)^T = B^T * A^T
    dim3 grid_coalesced_both((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    gpu_matmul_coalesced_both<<<grid_coalesced_both, block>>>(d_at, d_bt, d_ct, m, n, k);
    // теперь транспонируем результат, чтобы получить A*B
    dim3 grid_transpose_result((n + block_transpose.x - 1) / block_transpose.x,
                               (m + block_transpose.y - 1) / block_transpose.y);
    transpose_matrix<<<grid_transpose_result, block_transpose>>>(d_ct, d_c, n, m);
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float gpu_coalesced_both_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_coalesced_both_time, start_event, stop_event));

    // gpu shared
    CUDA_CHECK(cudaEventRecord(start_event));
    gpu_matmul_shared<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float gpu_shared_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_shared_time, start_event, stop_event));

    std::cout << "cpu time: " << cpu_time << " s\n";
    std::cout << "gpu naive time: " << gpu_naive_time / 1000.0f << " s\n"; // преобразуем миллисекунды в секунды
    std::cout << "gpu coalesced time: " << gpu_coalesced_time / 1000.0f << " s\n";
    std::cout << "gpu coalesced (both) time: " << gpu_coalesced_both_time / 1000.0f << " s\n";
    std::cout << "gpu shared time: " << gpu_shared_time / 1000.0f << " s\n";
    std::cout << "speedup (coalesced): " << cpu_time / (gpu_coalesced_time / 1000.0f) << "x\n";
    std::cout << "speedup (coalesced both): " << cpu_time / (gpu_coalesced_both_time / 1000.0f) << "x\n";
    std::cout << "speedup (shared): " << cpu_time / (gpu_shared_time / 1000.0f) << "x\n";

    // освобождение памяти
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_at);
    cudaFree(d_bt);
    cudaFree(d_ct);

    return 0;
}