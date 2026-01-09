### Как скомпилировать и запустить

```bash
nvcc sort.cu -o sort -O3
./sort
```

### Пояснение реализации

Для замера времени на CPU использовал библитечную сортировку (`sort`). Такое сравнение не совсем честное (в пользу CPU, так как библиотечная реализация оптимизированние ручной реализации), так как под капотом у этой сортировки не radix. Но даже в таком случае GPU побеждает с огромным отрывом.

Для эталона GPU использовал библиотечный thurst.

Сам реализовал 2 алгоритма сортировки. Их общий принцип следующий:
1) В цикле по количеству цифр (что такое цифра контролируется константой, в экспериментах система счисления имеет базу 256) в unsigned int повторить 2-5
2) Подсчёт кол-ва цифр в каждом блоке (map для получения цифры числа, atomicAdd для складывания в глобальную память)
3) Подсчёт префиксной суммы для каждого блока, то есть считаем сколько раз для каждого блока каждое число встречалось в предшестующих блоках (scan)
4) Подсчёт префиксной суммы для цифр (scan)
5) Запись числа исходного массива в ячейкку номер `digitShift + blockDigitShift + threadInBlockDigitShift`

2 алгоритма отличаются друг от друга scan-ом.

#### 1. Blelloch scan

![График NVIDIA](https://habrastorage.org/r/w1560/files/d17/7f0/106/d177f0106c9342a7bc6f6ae0d1b777b8.png)

#### 2. Hillis and Steele scan

![График NVIDIA](https://developer.download.nvidia.com/books/gpugems3/39fig02.jpg)

### Результаты

При росте массива наблюдается и рост разницы в скорости между GPU и CPU. Алгоритм radix сортировки при этом оказался быстрее подхода со сравнениями. Radix-сортировка на базе алгоритма scan от Hillis and Steele оказалась быстрее (в 1.5-3 раза), чем сортировка на базе Blelloch scan. При росте размера массива разница между сортировками на базе двух сканов сокращается.

```
@gpu:~/radix_sort$ ./sort_check
N:100000
Data Size: 100000
Padded Size (for Bitonic): 131072

1. Run CPU std::sort...
   Time: 0.0073 s

2. Run GPU Radix Sort (Blelloch Scan)...
   Time: 0.0007 s
   Status: Correct

3. Run GPU Radix Sort (Hillis-Steele Scan)...
   Time: 0.0002 s
   Status: Correct

4. Run GPU Bitonic Sort (Comparison Based, Padded input)...
   Time: 0.0007 s
   Status: Correct

@gpu:~/radix_sort$ ./sort_check
N:1000000
Data Size: 1000000
Padded Size (for Bitonic): 1048576

1. Run CPU std::sort...
   Time: 0.0861 s

2. Run GPU Radix Sort (Blelloch Scan)...
   Time: 0.0021 s
   Status: Correct

3. Run GPU Radix Sort (Hillis-Steele Scan)...
   Time: 0.0016 s
   Status: Correct

4. Run GPU Bitonic Sort (Comparison Based, Padded input)...
   Time: 0.0041 s
   Status: Correct

```

Результаты соответствуют (графикам от NVIDIA)[https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda]:

![График NVIDIA](https://developer.download.nvidia.com/books/gpugems3/39fig07.jpg)


### Хорошие документации:
- https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- https://habr.com/ru/companies/epam_systems/articles/247805/
