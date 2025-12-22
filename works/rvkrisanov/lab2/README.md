# Lab 2: Matrix Multiplication

## Задача
Реализовать и оптимизировать перемножение матриц на CUDA. Сравнить производительность с CPU.

## Реализация
1.  **CPU**: Классический алгоритм `O(N^3)`.
2.  **GPU Basic**: Наивная реализация (глобальная память).
3.  **GPU Tiled**: Оптимизация с Shared Memory (Tile 32x32) + Padding (устранение Bank Conflicts).

## Результаты (Turing GPU)

| Matrix Size | CPU (us) | CUDA Basic (us) | CUDA Tiled (us) | Speedup (Basic vs CPU) |
|---|---|---|---|---|
| 64x64 | 189 | 97 | 67 | 1.95x |
| 128x128 | 2,045 | 29 | 24 | 71x |
| 512x512 | 178,114 | 378 | 518 | 471x |
| 1024x1024 | 3,390,469 | 2,495 | 3,663 | 1358x |
| 2048x2048 | 37,524,274 | 17,843 | 27,324 | 2103x |

**Анализ:**
*   **Малые матрицы (64-128)**: Tiled версия быстрее за счет снижения латентности памяти.
*   **Большие матрицы (1024+)**: Basic версия оказывается быстрее.
    *   Причина: Современные GPU имеют эффективный L1/L2 кэш, который автоматически кэширует линейные доступы Basic-версии.
    *   Tiled версия имеет накладные расходы на `__syncthreads()` и инструкции управления индексами, которые на данной архитектуре перевешивают выигрыш от ручного управления памятью для простой операции FP32.

## Запуск
```bash
make
./matrix_multiply
```

```bash
(base) jovyan@a52120adfa7c:~/repos/cuda-cource-2025/works/rvkrisanov/lab2$ make
nvcc -O2 -arch=sm_70 -std=c++14 -Iinclude -o matrix_multiply src/main.cu src/matrix_multiply_cpu.cu src/matrix_multiply_cuda.cu src/matrix_multiply_cuda_tiled.cu
./matrix_multiply
Benchmarking Matrix 64x64...
  CPU:             189 us
  CUDA Basic:       97 us (1.95x speedup) [OK]
  CUDA Tiled:       67 us (2.81x speedup) [OK]

Benchmarking Matrix 128x128...
  CPU:            2045 us
  CUDA Basic:       29 us (71.17x speedup) [OK]
  CUDA Tiled:       24 us (85.44x speedup) [OK]

Benchmarking Matrix 256x256...
  CPU:           17304 us
  CUDA Basic:       69 us (249.19x speedup) [OK]
  CUDA Tiled:       80 us (216.30x speedup) [OK]

Benchmarking Matrix 512x512...
  CPU:          178114 us
  CUDA Basic:      378 us (471.30x speedup) [OK]
  CUDA Tiled:      518 us (344.18x speedup) [OK]

Benchmarking Matrix 1024x1024...
  CPU:         3390469 us
  CUDA Basic:     2495 us (1358.92x speedup) [OK]
  CUDA Tiled:     3663 us (925.52x speedup) [OK]

Benchmarking Matrix 2048x2048...
  CPU:        37524274 us
  CUDA Basic:    17843 us (2103.08x speedup) [OK]
  CUDA Tiled:    27324 us (1373.31x speedup) [OK]
```
