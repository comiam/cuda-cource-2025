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
cpu time: 2.414e-05 s
gpu naive time: 2.2528e-05 s
gpu coalesced time: 2.5888e-05 s
gpu coalesced (both) time: 2.2496e-05 s
gpu shared time: 1.6576e-05 s
speedup (coalesced): 0.932478x
speedup (coalesced both): 1.07308x
speedup (shared): 1.45632x
```

### Запуск 2 (средние матрицы: 512x512)

```
> enter matrix sizes m, n, k (e.g., 1024 1024 1024): 512 512 512
matrix size: 512x512 * 512x512
cpu time: 0.187761 s
gpu naive time: 0.000992256 s
gpu coalesced time: 0.00513635 s
gpu coalesced (both) time: 0.000951808 s
gpu shared time: 0.0007784 s
speedup (coalesced): 36.5554x
speedup (coalesced both): 197.268x
speedup (shared): 241.215x
```

### Запуск 3 (крупные матрицы: 1024x1024)

```
> enter matrix sizes m, n, k (e.g., 1024 1024 1024): 1024 1024 1024
matrix size: 1024x1024 * 1024x1024
cpu time: 3.73134 s
gpu naive time: 0.00374282 s
gpu coalesced time: 0.0203148 s
gpu coalesced (both) time: 0.00373462 s
gpu shared time: 0.00302976 s
speedup (coalesced): 183.677x
speedup (coalesced both): 999.122x
speedup (shared): 1231.56x
```

## Выводы

- Оптимизированная версия с использованием shared memory показывает наилучшие результаты
- Использование коалесцированного доступа к памяти улучшает производительность по сравнению с наивной реализацией
- Для больших матриц разница в производительности становится особенно заметной
