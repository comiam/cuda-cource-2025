## Lab 4: Radix Sort - Поразрядная сортировка

### Описание задачи
Реализовать алгоритм поразрядной сортировки (Radix Sort) на CUDA.

## Компиляция

```bash
make
```

## Запуск

```bash
./radix_sort
```

## Результаты

```
Radix Sort Benchmark
Array size: 1000000

init array with random data
Running CPU qsort
CPU time: 75.55 ms

Running GPU Radix Sort
GPU time: 8.37 ms
Speedup: 9.03x

Running Thrust sort
Thrust sort time: 1.07 ms

Verifying
Radix Sort: OK
Match with Thrust: OK
```
