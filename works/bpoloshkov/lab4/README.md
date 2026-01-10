# Lab 4: Radix Sort

## Описание

Реализация поразрядной сортировки (Radix Sort) на CUDA для знаковых 32-битных целых чисел.

## Требования

- CUDA Toolkit 11.0+
- GPU с поддержкой CUDA
- Make

## Компиляция и запуск

```bash
make
./radixsort [n]
```

Параметр `n` — размер массива (по умолчанию 1000000).

## Результаты

```
Array size: 1000000

CPU qsort time:     0.1434 sec
GPU Radix Sort:     0.0082 sec
GPU thrust::sort:   0.0009 sec

Verification: PASSED

Speedup vs CPU:     17.5x
```

## Алгоритм

4-битный LSD Radix Sort (8 проходов для 32-битных чисел).

Для каждого прохода (каждые 4 бита):

1. **countDigits** — подсчёт гистограммы через warp-примитивы
2. **prefixSumAndOffset** — вычисление позиций (все потоки активны)
3. **computeRanksAndScatter** — перестановка элементов через warp-примитивы

## Функции

| Функция | Назначение |
|---------|------------|
| `countDigits` | Подсчёт количества элементов с каждым разрядом. Warp-синхронизация через `__shfl_sync`. |
| `prefixSumAndOffset` | Exclusive prefix sum по блокам для вычисления глобальных позиций. |
| `computeRanksAndScatter` | Вычисление ранга через `__shfl_sync` и запись в выходной массив. |
| `toUnsigned` | Преобразование signed → unsigned через XOR 0x80000000. |
| `toSigned` | Обратное преобразование unsigned → signed. |
| `radixSortUnsigned` | Основной цикл сортировки: 8 проходов по 4 бита. |
| `radixSortSigned` | Обёртка для знаковых чисел. |

## Детали реализации

### Warp-примитивы через __shfl_sync


```cpp
unsigned int peersMask = 0;
for (int i = 0; i < 32; i++) {
    unsigned int otherDigit = __shfl_sync(0xFFFFFFFF, myDigit, i);
    if (otherDigit == myDigit) peersMask |= (1u << i);
}
```

- `__shfl_sync(mask, val, srcLane)` — получить значение `val` из потока `srcLane`
- `__popc(peersMask)` — количество потоков с тем же разрядом
- `__ffs(peersMask) - 1` — первый поток с данным разрядом (он записывает счётчик)

### Вычисление ранга внутри warp

```cpp
unsigned int rankInWarp = __popc(peersMask & ((1u << laneId) - 1));
```

Маска `(1u << laneId) - 1` выделяет биты потоков с меньшим laneId.

### Обработка знаковых чисел

XOR с 0x80000000 инвертирует знаковый бит:
- Отрицательные числа (бит 1) → меньшие unsigned
- Положительные числа (бит 0) → большие unsigned

### Параметры

- `BLOCK_SIZE = 256` — потоков на блок
- `RADIX_BITS = 4` — бит за проход (16 возможных значений)
- 8 проходов для 32-битных чисел

## Сложность

- Время: O(w/r * n), где w=32 бита, r=4 бита → O(8n)
- Память: O(RADIX_SIZE * numBlocks) для счётчиков
