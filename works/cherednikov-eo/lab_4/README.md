# Lab 4: Radix Sort - Поразрядная сортировка

## Описание

Реализация алгоритма поразрядной сортировки (Radix Sort) на CUDA для сортировки целых чисел различных типов (32-bit и 64-bit, signed и unsigned).

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
Benchmark: int32_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.13470s, GPU Thrust=1.71692s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.00x, Radix vs Thrust=12.75x

Benchmark: int64_t[1000] OK
Time: CPU=0.00002s, GPU Radix=0.02291s, GPU Thrust=0.00012s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.20x, Radix vs Thrust=0.01x

Benchmark: int32_t[100000] OK
Time: CPU=0.00416s, GPU Radix=0.01235s, GPU Thrust=0.00059s
Speedup: Radix vs CPU=0.34x, Thrust vs CPU=7.06x, Radix vs Thrust=0.05x

Benchmark: int64_t[100000] OK
Time: CPU=0.00435s, GPU Radix=0.02402s, GPU Thrust=0.00036s
Speedup: Radix vs CPU=0.18x, Thrust vs CPU=12.25x, Radix vs Thrust=0.01x

Benchmark: int32_t[5000000] OK
Time: CPU=0.27494s, GPU Radix=0.03907s, GPU Thrust=0.00167s
Speedup: Radix vs CPU=7.04x, Thrust vs CPU=164.98x, Radix vs Thrust=0.04x

Benchmark: int64_t[5000000] OK
Time: CPU=0.27699s, GPU Radix=0.08018s, GPU Thrust=0.00386s
Speedup: Radix vs CPU=3.45x, Thrust vs CPU=71.71x, Radix vs Thrust=0.05x

Benchmark: int32_t[10000000] OK
Time: CPU=0.57578s, GPU Radix=0.05052s, GPU Thrust=0.00306s
Speedup: Radix vs CPU=11.40x, Thrust vs CPU=187.89x, Radix vs Thrust=0.06x

Benchmark: int64_t[10000000] OK
Time: CPU=0.57905s, GPU Radix=0.10060s, GPU Thrust=0.00704s
Speedup: Radix vs CPU=5.76x, Thrust vs CPU=82.25x, Radix vs Thrust=0.07x

Benchmark: uint32_t[5000000] OK
Time: CPU=0.28323s, GPU Radix=0.03456s, GPU Thrust=0.00163s
Speedup: Radix vs CPU=8.20x, Thrust vs CPU=173.23x, Radix vs Thrust=0.05x

Benchmark: uint64_t[5000000] OK
Time: CPU=0.28232s, GPU Radix=0.06975s, GPU Thrust=0.00360s
Speedup: Radix vs CPU=4.05x, Thrust vs CPU=78.44x, Radix vs Thrust=0.05x
```