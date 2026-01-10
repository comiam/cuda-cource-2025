# **CUDA Lab 2 — Matrix Multiplication (CPU vs Naive CUDA vs Tiled CUDA)**

## 📁 Project Structure

```
Lab 2/
│   matmul_correctness.exe
│   benchmark_16.exe
│   benchmark_32.exe
│   benchmark_64.exe
│   ReadMe.md
│
├── src/
│    matmul_cpu.hpp
│    matmul_cpu.cpp
│    matmul_naive.cu
│    matmul_tiled.cu
│    main.cpp
│
└── benchmarks/
     benchmark.cpp
     results_16.csv
     results_32.csv
     results_64.csv
```

---

# ✔️ **1. Overview**

This project implements matrix multiplication on:

* **CPU (single-thread C++ implementation)**
* **Naive GPU kernel**
* **Tiled GPU kernel (shared memory + block tiling)**

Benchmarks include different tile sizes: **16, 32, 64**.

The project also includes correctness checks comparing CPU, naive GPU, and tiled GPU outputs.

---

# ✔️ **2. How to Build**

### CPU + GPU Build

```
nvcc -O3 src/main.cpp src/matmul_cpu.cpp src/matmul_naive.cu src/matmul_tiled.cu -o matmul_correctness.exe
```

### Benchmarks

```
nvcc -O3 -DTILE_DIM=16 -std=c++17 benchmarks/benchmark.cpp src/matmul_cpu.cpp src/matmul_naive.cu src/matmul_tiled.cu -o benchmark_16.exe
nvcc -O3 -DTILE_DIM=32 -std=c++17 benchmarks/benchmark.cpp src/matmul_cpu.cpp src/matmul_naive.cu src/matmul_tiled.cu -o benchmark_32.exe
nvcc -O3 -DTILE_DIM=64 -std=c++17 benchmarks/benchmark.cpp src/matmul_cpu.cpp src/matmul_naive.cu src/matmul_tiled.cu -o benchmark_64.exe
```

---
 # ✔️ **3. Correctness Checks**

Example output:

```
Test M=4 N=4 K=4
CPU:        0.0002 ms
Naive GPU:  0.2981 ms   [OK]
Tiled GPU:  0.3275 ms   [OK]

Test M=32 N=32 K=32
CPU:        0.0095 ms
Naive GPU:  0.3061 ms   [OK]
Tiled GPU:  0.2964 ms   [OK]

Test M=37 N=21 K=19
CPU:        0.0044 ms
Naive GPU:  0.4469 ms   [OK]
Tiled GPU:  0.3312 ms   [OK]

Test M=128 N=64 K=23
CPU:        0.0484 ms
Naive GPU:  0.3213 ms   [OK]
Tiled GPU:  0.3077 ms   [OK]

Test M=512 N=512 K=512
CPU:        67.2133 ms
Naive GPU:  1.3491 ms   [OK]
Tiled GPU:  1.297 ms   [OK]

Test M=1024 N=1024 K=1024
CPU:        2498.15 ms
Naive GPU:  6.0258 ms   [OK]
Tiled GPU:  5.6602 ms   [OK]


Press Enter to exit...
```

All tested sizes matched CPU results.

---

# ✔️ **4. Benchmark Results**

## 4.1 **Tile = 16**


| M    | N    | K    | CPU (ms)  | Naive Full (ms) | Naive Kernel (ms) | Tiled Full (ms) | Tiled Kernel (ms) |
|------|------|------|-----------|-----------------|-------------------|-----------------|-------------------|
| 32   | 32   | 32   | 0.009     | 0.300           | 0.021             | 0.298           | 0.017             |
| 128  | 128  | 128  | 0.855     | 0.322           | 0.022             | 0.324           | 0.022             |
| 256  | 256  | 256  | 8.757     | 0.489           | 0.087             | 0.458           | 0.059             |
| 512  | 512  | 512  | 66.338    | 1.337           | 0.484             | 1.246           | 0.390             |
| 1024 | 1024 | 1024 | 2458.560  | 5.690           | 3.659             | 5.027           | 3.031             |
| 512  | 1024 | 512  | 427.396   | 3.458           | 0.959             | 3.316           | 0.775             |
| 1024 | 512  | 1024 | 1153.848  | 5.289           | 1.840             | 4.971           | 1.521             |
| 512  | 1024 | 256  | 79.261    | 2.178           | 0.521             | 2.011           | 0.396             |


## 4.2 **Tile = 32**

| M    | N    | K    | CPU (ms)   | Naive Full (ms) | Naive Kernel (ms) | Tiled Full (ms) | Tiled Kernel (ms) |
|------|------|------|------------|-----------------|-------------------|-----------------|-------------------|
| 32   | 32   | 32   | 0.009      | 0.303           | 0.019             | 0.301           | 0.014             |
| 128  | 128  | 128  | 0.867      | 0.307           | 0.020             | 0.305           | 0.013             |
| 256  | 256  | 256  | 8.092      | 0.444           | 0.086             | 0.426           | 0.055             |
| 512  | 512  | 512  | 64.844     | 1.353           | 0.500             | 1.271           | 0.380             |
| 1024 | 1024 | 1024 | 3318.885   | 6.164           | 3.677             | 5.553           | 2.863             |
| 512  | 1024 | 512  | 451.078    | 3.422           | 0.962             | 3.343           | 0.729             |
| 1024 | 512  | 1024 | 1197.739   | 5.045           | 1.842             | 4.692           | 1.430             |
| 512  | 1024 | 256  | 80.865     | 2.335           | 0.521             | 2.236           | 0.369             |


## 4.3 **Tile = 64**

| M    | N    | K    | CPU (ms)   | Naive Full (ms) | Naive Kernel (ms) | Tiled Full (ms) | Tiled Kernel (ms) |
|------|------|------|------------|-----------------|-------------------|-----------------|-------------------|
| 32   | 32   | 32   | 0.010      | 0.284           | 0.020             | 0.268           | 0.007             |
| 128  | 128  | 128  | 0.820      | 0.310           | 0.020             | 0.292           | 0.012             |
| 256  | 256  | 256  | 8.382      | 0.471           | 0.086             | 0.350           | 0.054             |
| 512  | 512  | 512  | 172.955    | 1.337           | 0.484             | 0.758           | 0.363             |
| 1024 | 1024 | 1024 | 2938.057   | 7.033           | 3.695             | 3.560           | 2.825             |
| 512  | 1024 | 512  | 439.570    | 4.119           | 0.963             | 1.745           | 0.716             |
| 1024 | 512  | 1024 | 1210.123   | 4.480           | 1.842             | 1.671           | 1.426             |
| 512  | 1024 | 256  | 112.235    | 3.562           | 0.544             | 1.714           | 0.382             |


---

# ✔️ **5. Speedups**

## Formula

```
Speedup_CPU→GPU = CPU_time / GPU_kernel_time
Speedup_Naive→Tiled = naive_kernel / tiled_kernel
```

## Speedup Table

| M    | N    | K    | Tile | CPU → GPU Kernel Speedup | Naive → Tiled Kernel Speedup |
|------|------|------|------|--------------------------|------------------------------|
|      |      |      | 16   | 0.429                    | 1.235                        |
| 32   | 32   | 32   | 32   | 0.474                    | 1.357                        |
|      |      |      | 64   | 0.500                    | 2.857                        |
|      |      |      | 16   | 38.864                   | 1.000                        |
| 128  | 128  | 128  | 32   | 43.350                   | 1.538                        |
|      |      |      | 64   | 41.000                   | 1.667                        |
|      |      |      | 16   | 100.655                  | 1.475                        |
| 256  | 256  | 256  | 32   | 94.093                   | 1.564                        |
|      |      |      | 64   | 97.465                   | 1.593                        |
|      |      |      | 16   | 137.062                  | 1.241                        |
| 512  | 512  | 512  | 32   | 129.688                  | 1.316                        |
|      |      |      | 64   | 357.345                  | 1.333                        |
|      |      |      | 16   | 671.921                  | 1.207                        |
| 1024 | 1024 | 1024 | 32   | 902.607                  | 1.284                        |
|      |      |      | 64   | 795.144                  | 1.308                        |
|      |      |      | 16   | 445.668                  | 1.237                        |
| 512  | 1024 | 512  | 32   | 468.896                  | 1.320                        |
|      |      |      | 64   | 456.459                  | 1.345                        |
|      |      |      | 16   | 627.091                  | 1.210                        |
| 1024 | 512  | 1024 | 32   | 650.238                  | 1.288                        |
|      |      |      | 64   | 656.961                  | 1.292                        |
|      |      |      | 16   | 152.132                  | 1.316                        |
| 512  | 1024 | 256  | 32   | 155.211                  | 1.412                        |
|      |      |      | 64   | 206.314                  | 1.424                        |

## Best Tile Dimention for Speedup Table 

| M    | N    | K    | Best Tile (Naive → Tiled Kernel) |
|------|------|------|----------------------------------|
| 32   | 32   | 32   | 64                               |
| 128  | 128  | 128  | 64                               |
| 256  | 256  | 256  | 64                               |
| 512  | 512  | 512  | 64                               |
| 1024 | 1024 | 1024 | 64                               |
| 512  | 1024 | 512  | 64                               |
| 1024 | 512  | 1024 | 64                               |
| 512  | 1024 | 256  | 64                               |

---

# ✔️ **6. Conclusion**

The project implemented and compared three approaches to matrix multiplication: a CPU baseline, a naive CUDA kernel, and a tiled CUDA kernel using shared memory. All implementations produced correct results across multiple matrix sizes.

Benchmarking confirmed that the GPU provides significant acceleration compared to the CPU, especially for larger matrices where the high compute density fully utilizes the GPU’s parallel architecture. Across all tested dimensions, GPU kernel time was consistently orders of magnitude faster than CPU execution.

For this specific hardware and implementation, the naive kernel was very close to the tiled version. Profiling results indicate that matrix sizes may still be too small to benefit from shared-memory tiling, and that the tiled kernel incurs more overhead per block for loading shared tiles. Larger models—or more optimized kernels with loop unrolling, reduced register pressure, and carefully tuned block sizes—are expected to allow the tiled approach to surpass the naive implementation even more.

Tile size **64** showed the strongest speedups for all matrices, suggesting that this configuration achieves a good balance between occupancy and memory reuse on the tested GPU.

Overall, the lab demonstrates the full workflow of GPU performance engineering: implementing CPU/GPU kernels, benchmarking, profiling with CUDA events, and analyzing memory- vs compute-bound behavior. The results also highlight that optimization benefits depend strongly on GPU architecture, tile dimension, and workload size.

---