# Lab 4: Radix Sort - Поразрядная сортировка

## Описание задачи

Реализовать алгоритм поразрядной сортировки (Radix Sort) на CUDA.


## Требования
- CUDA Toolkit (`nvcc`) установлен
- `make` (для удобства сборки)

---

## Запуск и примеры

### Запуск

Сборка кода:

```bash
make
```
Запуск кода:

```bash
./bin/lab4
```

### Примеры

```md
Array size: 1000000

CPU sort time: 0.14 s
GPU Radix Sort time: 0.005 s
GPU thrust::sort time: 0.001 s
Speedup (Radix vs CPU): 26.3x
Speedup (thrust::sort vs CPU): 171.1x
```