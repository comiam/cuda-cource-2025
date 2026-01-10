### Как скомпилировать и запустить

```bash
nvcc sort.cu -o sort -O3
./sort
```

### Пояснение реализации

Для замера времени на CPU использовал библитечную сортировку (`sort`). Такое сравнение не совсем честное (в пользу CPU, так как библиотечная реализация оптимизированние ручной реализации), так как под капотом у этой сортировки не radix. Но даже в таком случае GPU побеждает на порядки.

Для эталона GPU использовал библиотечный thurst.

Сам реализовал 4 алгоритма сортировки. Один из них (bitonics) - на базе сравнений элементов. 3 других radix-сортивровок общий принцип следующий:
1) В цикле по количеству цифр (что такое цифра контролируется константой, в экспериментах система счисления имеет базу 256) в unsigned int повторить 2-5
2) Подсчёт кол-ва цифр в каждом блоке (map для получения цифры числа, atomicAdd для складывания в глобальную память)
3) Подсчёт префиксной суммы для каждого блока, то есть считаем сколько раз для каждого блока каждое число встречалось в предшестующих блоках (scan)
4) Подсчёт префиксной суммы для цифр (scan)
5) Запись числа исходного массива в ячейкку номер `digitShift + blockDigitShift + threadInBlockDigitShift`
3

3 алгоритма отличаются друг от друга scan-ом.

#### 1. Blelloch scan

Я реализовал 2 версии:
- Строит гистограмму с помощью atomicAdd в shared memory
- Сторит гистограмму, синхронизируясь в рамках варпа

![График NVIDIA](https://habrastorage.org/r/w1560/files/d17/7f0/106/d177f0106c9342a7bc6f6ae0d1b777b8.png)

#### 2. Hillis and Steele scan

![График NVIDIA](https://developer.download.nvidia.com/books/gpugems3/39fig02.jpg)

### Результаты

При росте массива наблюдается и рост разницы в скорости между GPU и CPU. Алгоритм radix сортировки при этом оказался быстрее подхода со сравнениями. Radix-сортировка на базе алгоритма scan от Hillis and Steele оказалась немного быстрее, чем сортировка на базе Blelloch scan (в atomic и не atomic версиях). Время работы Blelloch с atomic и с варп синхронизацией примерно равны.

```
@gpu:~/radix_sort$ ./sort
N:100000
Data Size: 100000
Padded Size (for Bitonic): 131072

1. Run CPU std::sort...
   Time: 0.0071 s

2. Run Thrust Sort...
   Time: 0.0032 s
   Status: Correct

3. Run GPU Radix Sort (Hillis-Steele + Atomics)...
      [Kernel Logs] Hist: 0.0496ms, Scan: 0.0558ms, Scatter: 0.1565ms
   Time: 0.0004 s
   Status: Correct

4. Run GPU Radix Sort (Blelloch + Atomics)...
      [Kernel Logs] Hist: 0.0361ms, Scan: 0.1696ms, Scatter: 0.1395ms
   Time: 0.0005 s
   Status: Correct

5. Run GPU Radix Sort (Blelloch + Warp Sync No Atomics)...
      [Kernel Logs] Hist: 0.0990ms, Scan: 0.1562ms, Scatter: 0.1416ms
   Time: 0.0005 s
   Status: Correct

6. Run GPU Bitonic Sort...
   Time: 0.0007 s
   Status: Correct

@gpu:~/radix_sort$ ./sort
N:1000000
Data Size: 1000000
Padded Size (for Bitonic): 1048576

1. Run CPU std::sort...
   Time: 0.0845 s

2. Run Thrust Sort...
   Time: 0.0042 s
   Status: Correct

3. Run GPU Radix Sort (Hillis-Steele + Atomics)...
      [Kernel Logs] Hist: 0.2421ms, Scan: 0.1946ms, Scatter: 1.3736ms
   Time: 0.0022 s
   Status: Correct

4. Run GPU Radix Sort (Blelloch + Atomics)...
      [Kernel Logs] Hist: 0.2360ms, Scan: 0.2228ms, Scatter: 1.3567ms
   Time: 0.0022 s
   Status: Correct

5. Run GPU Radix Sort (Blelloch + Warp Sync No Atomics)...
      [Kernel Logs] Hist: 0.7010ms, Scan: 0.2087ms, Scatter: 1.3553ms
   Time: 0.0026 s
   Status: Correct

6. Run GPU Bitonic Sort...
   Time: 0.0051 s
   Status: Correct

```

Результаты соответствуют (графикам от NVIDIA)[https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda]:

![График NVIDIA](https://developer.download.nvidia.com/books/gpugems3/39fig07.jpg)


### Хорошие документации:
- https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- https://habr.com/ru/companies/epam_systems/articles/247805/
