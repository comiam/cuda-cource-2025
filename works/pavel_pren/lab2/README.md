# Lab 2: Matrix Multiplication with CUDA

## Файлы

- `matrix_multiply.cu` - основная программа с CPU и GPU реализациями умножения матриц

## Компиляция

```bash
nvcc -o matrix_multiply matrix_multiply.cu
```

## Запуск

```bash
./matrix_multiply
```

## Результаты
128x128 matrices
CPU Time: 0.0246 sec
GPU basic Time: 0.0001 sec, result correct
GPU with tiling + shared memory Time: 0.0001 sec, result correct
Speedup (basic vs CPU): 216.94x
Speedup (tiled vs CPU): 335.62x
Speedup (tiled vs basic): 1.55x

32x32 matrices
CPU Time: 0.0001 sec
GPU basic Time: 0.0041 sec, result correct
GPU with tiling + shared memory Time: 0.0016 sec, result correct
Speedup (basic vs CPU): 0.01x
Speedup (tiled vs CPU): 0.03x
Speedup (tiled vs basic): 2.60x

128x256 × 256x128 matrices
CPU Time: 0.0542 sec
GPU basic Time: 0.0001 sec, result correct
GPU with tiling + shared memory Time: 0.0001 sec, result correct
Speedup (basic vs CPU): 869.57x
Speedup (tiled vs CPU): 701.86x
Speedup (tiled vs basic): 0.81x

512x512 matrices
CPU Time: 1.0934 sec
GPU basic Time: 0.0003 sec, result correct
GPU with tiling + shared memory Time: 0.0002 sec, result correct
Speedup (basic vs CPU): 3920.11x
Speedup (tiled vs CPU): 6046.31x
Speedup (tiled vs basic): 1.54x

1024x1024 matrices
CPU Time: 8.2782 sec
GPU basic Time: 0.0013 sec, result correct
GPU with tiling + shared memory Time: 0.0008 sec, result correct
Speedup (basic vs CPU): 6563.69x
Speedup (tiled vs CPU): 10047.56x
Speedup (tiled vs basic): 1.53x

2048x2048 matrices
CPU Time: 61.5722 sec
GPU basic Time: 0.0943 sec, result correct
GPU with tiling + shared memory Time: 0.0590 sec, result correct
Speedup (basic vs CPU): 653.11x
Speedup (tiled vs CPU): 1043.27x
Speedup (tiled vs basic): 1.60x

4096x4096 matrices
CPU Time: 365.1092 sec
GPU basic Time: 0.5180 sec, result correct
GPU with tiling + shared memory Time: 0.0427 sec, result correct
Speedup (basic vs CPU): 704.84x
Speedup (tiled vs CPU): 8551.08x
Speedup (tiled vs basic): 12.13x