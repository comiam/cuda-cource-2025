# Lab 4: Parallel Radix Sort

## Задача
Реализовать параллельную сортировку Radix Sort (LSD) на CUDA без использования сторонних библиотек для логики алгоритма.
Поддержка `int32`, `int64` и отрицательных чисел.

## Реализация
Использован **Binary Radix Sort** (поразрядная сортировка по 1 биту).
1.  **Predicate Kernel**: Формирование битовой маски (0/1).
2.  **Scan (Prefix Sum)**: Реализован собственный рекурсивный алгоритм (Blelloch) для вычисления индексов.
3.  **Scatter Kernel**: Перестановка элементов на основе скана.
4.  **Radix Hack**: Инверсия знакового бита для поддержки отрицательных чисел.

## Результаты
Сравнение на массивах 10^5 - 10^6 элементов.

### Int32 (32-bit)
| N | CPU (us) | Custom GPU (us) | Thrust (us) | Speedup (vs CPU) |
|---|---|---|---|---|
| 100k | 5,313 | 1,274 | 103 | **4.17x** |
| 500k | 29,742 | 2,606 | 350 | **11.41x** |
| 1M | 62,460 | 4,270 | 420 | **14.63x** |

### Int64 (64-bit)
| N | CPU (us) | Custom GPU (us) | Thrust (us) | Speedup (vs CPU) |
|---|---|---|---|---|
| 100k | 5,250 | 2,572 | 149 | **2.04x** |
| 500k | 30,109 | 5,432 | 499 | **5.54x** |
| 1M | 62,875 | 9,329 | 698 | **6.74x** |

**Вывод:** Реализация корректна. Достигнуто ускорение до 14.6x относительно CPU. Собственный алгоритм медленнее библиотечного Thrust из-за множественных обращений к глобальной памяти (Memory Bound) и отсутствия advanced-оптимизаций (warp shuffle scans).

## Запуск
```bash
make
make benchmark  # Запуск тестов
```

```bash
Warming up CUDA and Thrust...
Warmup done.
================================================================
Running Benchmark for Type: int32 (32-bit)
================================================================
Size N=100000
Verifying Custom Implementation: [OK] Results match!
Avg Time CPU:           5313 us
Avg Time Custom GPU:    1274 us
Avg Time Thrust GPU:    103 us
Speedup vs CPU:     4.16762x
Speedup vs Thrust:  0.0814691x
------------------------------------------------
Size N=500000
Verifying Custom Implementation: [OK] Results match!
Avg Time CPU:           29742 us
Avg Time Custom GPU:    2606 us
Avg Time Thrust GPU:    350 us
Speedup vs CPU:     11.4128x
Speedup vs Thrust:  0.134315x
------------------------------------------------
Size N=1000000
Verifying Custom Implementation: [OK] Results match!
Avg Time CPU:           62460 us
Avg Time Custom GPU:    4270 us
Avg Time Thrust GPU:    420 us
Speedup vs CPU:     14.6267x
Speedup vs Thrust:  0.0985104x
------------------------------------------------


================================================================
Running Benchmark for Type: int64 (64-bit)
================================================================
Size N=100000
Verifying Custom Implementation: [OK] Results match!
Avg Time CPU:           5250 us
Avg Time Custom GPU:    2572 us
Avg Time Thrust GPU:    149 us
Speedup vs CPU:     2.04073x
Speedup vs Thrust:  0.0579347x
------------------------------------------------
Size N=500000
Verifying Custom Implementation: [OK] Results match!
Avg Time CPU:           30109 us
Avg Time Custom GPU:    5432 us
Avg Time Thrust GPU:    499 us
Speedup vs CPU:     5.54237x
Speedup vs Thrust:  0.0919865x
------------------------------------------------
Size N=1000000
Verifying Custom Implementation: [OK] Results match!
Avg Time CPU:           62875 us
Avg Time Custom GPU:    9329 us
Avg Time Thrust GPU:    698 us
Speedup vs CPU:     6.73946x
Speedup vs Thrust:  0.0748452x
------------------------------------------------
```
