## Lab 2: Matrix Multiplication - Перемножение матриц

### Описание задачи
Реализовать перемножение матриц на CUDA.
---

## Требования
- CUDA Toolkit (`nvcc`) установлен
- Компьютер с поддержкой CUDA (NVIDIA GPU)
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
./lab2
```

### Примеры

```md
Matrix size: 1024x1024
CPU time: 2.476 s
GPU time (basic): 0.004 s
GPU time (tiled/shared): 0.003 s    
Speed UP: ~567.0x
Speed UP (tiled): ~869.9x
```