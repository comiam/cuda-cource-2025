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
CPU:        0.0016 ms
Naive GPU:  0.3773 ms   [OK]
Tiled GPU:  0.4058 ms   [OK]

Test M=32 N=32 K=32
CPU:        0.0528 ms
Naive GPU:  0.308 ms   [OK]
Tiled GPU:  0.3318 ms   [OK]

Test M=37 N=21 K=19
CPU:        0.0246 ms
Naive GPU:  0.2717 ms   [OK]
Tiled GPU:  0.399 ms   [OK]

Test M=128 N=64 K=23
CPU:        0.3241 ms
Naive GPU:  0.4148 ms   [OK]
Tiled GPU:  0.3376 ms   [OK]

Test M=512 N=512 K=512
CPU:        256.364 ms
Naive GPU:  1.3676 ms   [OK]
Tiled GPU:  1.3881 ms   [OK]

Test M=1024 N=1024 K=1024
CPU:        2857.37 ms
Naive GPU:  5.0925 ms   [OK]
Tiled GPU:  5.6833 ms   [OK]


Press Enter to exit...
```

All tested sizes matched CPU results.

---

# ✔️ **4. Benchmark Results**

## 4.1 **Tile = 16**

<!-- FILL with  your actual numbers from results_16.csv -->


| M | N | K | CPU (ms) | Naive Full (ms) | Naive Kernel (ms) | Tiled Full (ms) | Tiled Kernel (ms) |
|---|---|---|----------|-----------------|-------------------|-----------------|-------------------|
| 32   | 32   | 32   | 0.009    | 0.293 | 0.018 | 0.289 | 0.009 |
| 128  | 128  | 128  | 0.874    | 0.328 | 0.020 | 0.332 | 0.018 |
| 256  | 256  | 256  | 8.399    | 0.440 | 0.053 | 0.445 | 0.059 |
| 512  | 512  | 512  | 71.790   | 1.242 | 0.350 | 1.287 | 0.391 |
| 1024 | 1024 | 1024 | 5359.715 | 5.510 | 2.716 | 5.785 | 3.051 |


## 4.2 **Tile = 32**

| M | N | K | CPU (ms) | Naive Full (ms) | Naive Kernel (ms) | Tiled Full (ms) | Tiled Kernel (ms) |
|---|---|---|----------|-----------------|-------------------|-----------------|-------------------|
| 32   | 32   | 32   | 0.010    | 0.291 | 0.007 | 0.304 | 0.008 |
| 128  | 128  | 128  | 0.958    | 0.319 | 0.013 | 0.316 | 0.013 |
| 256  | 256  | 256  | 8.181    | 0.459 | 0.053 | 0.486 | 0.055 |
| 512  | 512  | 512  | 67.534   | 1.242 | 0.356 | 1.302 | 0.374 |
| 1024 | 1024 | 1024 | 2591.358 | 4.846 | 2.699 | 5.366 | 2.851 |

## 4.3 **Tile = 64**

| M | N | K | CPU (ms) | Naive Full (ms) | Naive Kernel (ms) | Tiled Full (ms) | Tiled Kernel (ms) |
|---|---|---|----------|-----------------|-------------------|-----------------|-------------------|
| 32   | 32   | 32   | 0.007    | 0.294 | 0.017 | 0.283 | 0.008 |
| 128  | 128  | 128  | 0.864    | 0.316 | 0.015 | 0.293 | 0.014 |
| 256  | 256  | 256  | 8.266    | 0.439 | 0.053 | 0.349 | 0.055 |
| 512  | 512  | 512  | 68.904   | 1.235 | 0.376 | 0.819 | 0.394 |
| 1024 | 1024 | 1024 | 2647.899 | 4.906 | 2.703 | 1.907 | 2.829 |

---

# ✔️ **5. Speedups**

## Formula

```
Speedup_CPU→GPU = CPU_time / GPU_kernel_time
Speedup_Naive→Tiled = naive_kernel / tiled_kernel
```

## Speedup Table

| M | N | K | Tile | CPU → GPU Kernel Speedup | Naive → Tiled Kernel Speedup |
|---|---|---|------|--------------------------|------------------------------|
| 32   | 32   | 32   | 16 | 0.500    | 2.000 |
| 32   | 32   | 32   | 32 | 1.429    | 0.875 |
| 32   | 32   | 32   | 64 | 0.412    | 2.125 |
| 128  | 128  | 128  | 16 | 43.700   | 1.111 |
| 128  | 128  | 128  | 32 | 73.692   | 1.000 |
| 128  | 128  | 128  | 64 | 57.600   | 1.071 |
| 256  | 256  | 256  | 16 | 158.472  | 0.898 |
| 256  | 256  | 256  | 32 | 154.358  | 0.964 |
| 256  | 256  | 256  | 64 | 155.962  | 0.964 |
| 512  | 512  | 512  | 16 | 205.114  | 0.895 |
| 512  | 512  | 512  | 32 | 189.702  | 0.952 |
| 512  | 512  | 512  | 64 | 183.255  | 0.954 |
| 1024 | 1024 | 1024 | 16 | 1973.385 | 0.890 |
| 1024 | 1024 | 1024 | 32 | 960.118  | 0.947 |
| 1024 | 1024 | 1024 | 64 | 979.615  | 0.955 |

## Best Tile Dimention for Speedup Table 

| M | N | K | Best Tile (Naive → Tiled Kernel) |
|---|---|---|----------------------------------|
| 32   | 32   | 32   | 64 |
| 128  | 128  | 128  | 16 |
| 256  | 256  | 256  | 32/64 |
| 512  | 512  | 512  | 64 |
| 1024 | 1024 | 1024 | 64 |

---

# ✔️ **6. Conclusion**

The project implemented and compared three approaches to matrix multiplication: a CPU baseline, a naive CUDA kernel, and a tiled CUDA kernel using shared memory. All implementations produced correct results across multiple matrix sizes.

Benchmarking confirmed that the GPU provides significant acceleration compared to the CPU, especially for larger matrices where the high compute density fully utilizes the GPU’s parallel architecture. Across all tested dimensions, GPU kernel time was consistently orders of magnitude faster than CPU execution.

For this specific hardware and implementation, the naive kernel often outperformed the tiled version. Profiling results indicate that matrix sizes may still be too small to benefit from shared-memory tiling, and that the tiled kernel incurs more overhead per block for loading shared tiles. Larger models—or more optimized kernels with loop unrolling, reduced register pressure, and carefully tuned block sizes—are expected to allow the tiled approach to surpass the naive implementation, as described in NVIDIA’s optimization guidelines.

Tile size **64** showed the strongest speedups for large matrices (512–1024), suggesting that this configuration achieves a good balance between occupancy and memory reuse on the tested GPU.

Overall, the lab demonstrates the full workflow of GPU performance engineering: implementing CPU/GPU kernels, benchmarking, profiling with CUDA events, and analyzing memory- vs compute-bound behavior. The results also highlight that optimization benefits depend strongly on GPU architecture, tile dimension, and workload size.

---