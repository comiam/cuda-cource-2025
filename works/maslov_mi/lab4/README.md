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

#### Array size: 100000
```
Array size: 100000

CPU sort time: 0.0036 s
GPU Radix Sort time: 0.0016 s
GPU thrust::sort time: 0.0008 s
Speedup (Radix vs CPU): 2.32x
Speedup (thrust::sort vs CPU): 4.71x
Speedup (thrust::sort vs Radix): 2.03x
```

#### Array size: 1000000
```
Array size: 1000000

CPU sort time: 0.0421 s
GPU Radix Sort time: 0.0045 s
GPU thrust::sort time: 0.0015 s
Speedup (Radix vs CPU): 9.28x
Speedup (thrust::sort vs CPU): 28.78x
Speedup (thrust::sort vs Radix): 3.10x
```

#### Array size: 10000000
```
Array size: 10000000

CPU sort time: 0.4836 s
GPU Radix Sort time: 0.0191 s
GPU thrust::sort time: 0.0034 s
Speedup (Radix vs CPU): 25.38x
Speedup (thrust::sort vs CPU): 144.12x
Speedup (thrust::sort vs Radix): 5.68x
```