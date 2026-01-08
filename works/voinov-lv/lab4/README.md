## Лабораторная 4

### Описание задачи

Реализовать алгоритм поразрядной сортировки (Radix Sort) на CUDA.

### Требования

- Базовая реализация Radix Sort на GPU
- Корректная сортировка тестовых данных
- Сортировка целых чисел (32-bit и 64-bit)
- Поддержка массивов размером от 10^5 до 10^6 элементов
- Обработка как положительных, так и отрицательных чисел
- Бенчмарк сравнения с CPU и thrust

### Компиляция

```make```

### Запуск

```./main```

### Описание реализации

Реализация алгоритма Radix Sort на CUDA использует для работы несколько различных ядер.

Ядро preds_kernel вычисляет инвертированные битовые предикаты для сортировки нулей раньше единиц (битовая маска).

Ядро prescan_kernel реализует параллельную префиксную сумму внутри блока по алгоритму Blelloch.

Функция exclusive_scan считает глобальную префиксную сумму с помощью рекурсивной обработки блоков.

Ядро add_block_sums_kernel выполняет поэлементное сложение каждого блока с его накопленным смещением.

Ядро scatter_kernel перераспределяет элементы по новым позициям на основе префиксных сумм.

Основная функция radix_sort координирует сортировку, последовательно запуская ядра для каждого бита.

### Вывод программы

```
100000 32-bit integers
CPU sort time: 0.0052738 sec
GPU Radix Sort time: 0.00325402 sec
GPU thrust::sort time: 0.000601312 sec
GPU Radix Sort is 1.62071x faster than CPU
GPU thrust::sort is 8.77049x faster than CPU
Radix Sort: Success
Thrust Sort: Success

1000000 32-bit integers
CPU sort time: 0.0648007 sec
GPU Radix Sort time: 0.0214096 sec
GPU thrust::sort time: 0.00187456 sec
GPU Radix Sort is 3.02671x faster than CPU
GPU thrust::sort is 34.5685x faster than CPU
Radix Sort: Success
Thrust Sort: Success

100000000 32-bit integers
CPU sort time: 8.92453 sec
GPU Radix Sort time: 0.91658 sec
GPU thrust::sort time: 0.127106 sec
GPU Radix Sort is 9.73677x faster than CPU
GPU thrust::sort is 70.2135x faster than CPU
Radix Sort: Success
Thrust Sort: Success

100000 64-bit integers
CPU sort time: 0.0059172 sec
GPU Radix Sort time: 0.00481267 sec
GPU thrust::sort time: 0.000912416 sec
GPU Radix Sort is 1.2295x faster than CPU
GPU thrust::sort is 6.4852x faster than CPU
Radix Sort: Success
Thrust Sort: Success

1000000 64-bit integers
CPU sort time: 0.0706935 sec
GPU Radix Sort time: 0.0350948 sec
GPU thrust::sort time: 0.00290694 sec
GPU Radix Sort is 2.01436x faster than CPU
GPU thrust::sort is 24.3188x faster than CPU
Radix Sort: Success
Thrust Sort: Success

100000000 64-bit integers
CPU sort time: 9.66188 sec
GPU Radix Sort time: 2.46232 sec
GPU thrust::sort time: 0.267559 sec
GPU Radix Sort is 3.92389x faster than CPU
GPU thrust::sort is 36.1112x faster than CPU
Radix Sort: Success
Thrust Sort: Success
```

### Результаты

Программа реализует Radix Sort на GPU и проводит сравнение скорости сортировки на 100000, 1000000 и 100000000 32-битных и 64-битных целых числах. Все сортировки работают корректно. Thrust::sort - наиболее быстрая сортировка, затем следует Radix, и самая медленная сортировка на CPU. Чем больше массив, тем большее ускорение дают сортировки на GPU относительно CPU.
