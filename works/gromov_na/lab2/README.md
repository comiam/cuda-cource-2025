# CUDA Matrix Multiplication

Реализация умножения матриц на CUDA с различными подходами к оптимизации.

## Основные особенности

- Умножение матриц A[m,k] \* B[k,n] = C[m,n]
- CPU реализация для сравнения
- Наивная GPU реализация
- GPU реализация с коалесцированным доступом к одной матрице
- GPU реализация с коалесцированным доступом к обеим матрицам
- GPU реализация с использованием shared memory

## Результаты тестирования

### Запуск 1 (малые матрицы: 32x32)

```
> enter matrix sizes m, n, k (e.g., 1024 1024 1024): 32 32 32
matrix size: 32x32 * 32x32
cpu time: 0.000104386 s
gpu naive time: 1.8484e-05 s
gpu coalesced time: 2.2087e-05 s
gpu coalesced (both) time: 2.8418e-05 s
gpu shared time: 1.9579e-05 s
speedup (coalesced): 4.72613x
speedup (coalesced both): 3.67324x
speedup (shared): 5.33153x
```

### Запуск 2 (средние матрицы: 512x512)

```
> enter matrix sizes m, n, k (e.g., 1024 1024 1024): 512 512 512
matrix size: 512x512 * 512x512
cpu time: 0.473271 s
gpu naive time: 0.000503631 s
gpu coalesced time: 0.00242976 s
gpu coalesced (both) time: 0.00243617 s
gpu shared time: 0.000294271 s
speedup (coalesced): 194.781x
speedup (coalesced both): 194.268x
speedup (shared): 1608.28x
```

### Запуск 3 (крупные матрицы: 1024x1024)

```
> enter matrix sizes m, n, k (e.g., 1024 1024 1024): 1024 1024 1024
matrix size: 1024x1024 * 1024x1024
cpu time: 6.0976 s
gpu naive time: 0.00928773 s
gpu coalesced time: 0.0520228 s
gpu coalesced (both) time: 0.0518853 s
gpu shared time: 0.00581681 s
speedup (coalesced): 117.21x
speedup (coalesced both): 117.521x
speedup (shared): 1048.27x
```

## Выводы

- Оптимизированная версия с использованием shared memory показывает наилучшие результаты
- Использование коалесцированного доступа к памяти улучшает производительность по сравнению с наивной реализацией
- Для больших матриц разница в производительности становится особенно заметной
