# Lab 2: Matrix Multiplication

## Задача
Реализовать и оптимизировать перемножение матриц на CUDA. Сравнить производительность с CPU.

## Реализация
1.  **CPU**: Классический алгоритм `O(N^3)`.
2.  **GPU Basic**: Наивная реализация (глобальная память).
3.  **GPU Tiled**: Оптимизация с Shared Memory (Tile 16x16) + Unrolling.

## Результаты (Turing GPU)

| Matrix Size | CPU (us) | CUDA Basic (us) | CUDA Tiled (us) | Speedup (Basic vs CPU) |
|---|---|---|---|---|
| 64x64 | 333 | 166 | 116 | 2.00x |
| 128x128 | 3,405 | 23 | 24 | 145x |
| 512x512 | 209,305 | 384 | 518 | 545x |
| 1024x1024 | 3,389,726 | 2,466 | 3,709 | 1374x |
| 2048x2048 | 38,546,840 | 17,936 | 28,240 | 2149x |

**Анализ:**
*   **Малые матрицы**: Tiled версия показывает себя хорошо или на уровне Basic.
*   **Большие матрицы (1024+)**: Basic версия оказывается быстрее.
    *   Причина: Современные GPU (Turing/Ampere) имеют крайне эффективный L1/L2 кэш, который автоматически кэширует линейные доступы Basic-версии.
    *   Tiled версия имеет накладные расходы на `__syncthreads()` и более сложную арифметику (вычисление индексов, загрузка в shared memory). Без использования продвинутых техник (prefetching, vectorization) выигрыш от shared memory нивелируется этими расходами на данной архитектуре.

## Запуск
```bash
make
./matrix_multiply
```

```bash
(base) jovyan@a52120adfa7c:~/workspace/CUDAIAPPROACHED/cuda-cource-2025/works/rvkrisanov/lab2$ make
nvcc -O2 -arch=sm_70 -std=c++14 -Iinclude -o matrix_multiply src/main.cu src/matrix_multiply_cpu.cu src/matrix_multiply_cuda.cu src/matrix_multiply_cuda_tiled.cu
./matrix_multiply
Benchmarking Matrix 64x64...
  CPU:             333 us
  CUDA Basic:      166 us (2.00x speedup) [OK]
  CUDA Tiled:      116 us (2.88x speedup) [OK]

Benchmarking Matrix 128x128...
  CPU:            3405 us
  CUDA Basic:       23 us (145.56x speedup) [OK]
  CUDA Tiled:       24 us (140.19x speedup) [OK]

Benchmarking Matrix 256x256...
  CPU:           29691 us
  CUDA Basic:       83 us (357.83x speedup) [OK]
  CUDA Tiled:       79 us (373.98x speedup) [OK]

Benchmarking Matrix 512x512...
  CPU:          209305 us
  CUDA Basic:      384 us (545.57x speedup) [OK]
  CUDA Tiled:      518 us (403.85x speedup) [OK]

Benchmarking Matrix 1024x1024...
  CPU:         3389726 us
  CUDA Basic:     2466 us (1374.65x speedup) [OK]
  CUDA Tiled:     3709 us (913.94x speedup) [OK]

Benchmarking Matrix 2048x2048...
  CPU:        38546840 us
  CUDA Basic:    17936 us (2149.17x speedup) [OK]
  CUDA Tiled:    28240 us (1364.99x speedup) [OK]
```
