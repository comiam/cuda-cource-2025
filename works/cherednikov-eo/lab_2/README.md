# Лаюораторная работа №2

### Основные функции:

- cpu_matmul - наивная CPU-функция
- gpu_naive_mm - наивная GPU-функция
- gpu_optimised_mm - оптимизированная GPU функция

### Результаты:

- Матрицы 32x48 48x32:

GPU naive      time: 0.098 ms
GPU optimized  time: 0.016 ms
CPU time: 0.088 ms
Results are correct. GPU speedup = 5.977

- Матрицы 2048x2048:

GPU naive      time: 20.129 ms
GPU optimized  time: 13.109 ms
CPU time: 10413.236 ms
Results are correct. GPU speedup = 0.651

- Матрицы 4096x4096:

GPU naive      time: 153.781 ms
GPU optimized  time: 90.911 ms
CPU time: 86673.565 ms
Results are correct. GPU speedup = 0.591