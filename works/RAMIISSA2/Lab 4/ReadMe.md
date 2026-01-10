
# CUDA Laboratory Work №4 — GPU Radix Sort

---

## 1. Task Description

The objective of this laboratory work is to **implement and analyze a GPU-based Radix Sort algorithm using CUDA**. The work focuses on developing a custom parallel Radix Sort for signed integers and comparing its performance and correctness against:

1. CPU-based sorting algorithms:

   * `std::sort`
   * `qsort`
2. A GPU reference implementation:

   * `thrust::sort`

The primary goal is **not only performance comparison**, but also a **practical study of parallel scan-based algorithms**, kernel launch overhead, and memory access behavior on modern GPU architectures.

---

## 2. Radix Sort Algorithm Overview

Radix Sort is a **non-comparative sorting algorithm** that processes integers **digit-by-digit** (or bit-by-bit). In this laboratory work, a **Least Significant Bit (LSB) Radix Sort** is implemented, where the input array is processed over **32 passes**, one per bit.

For each bit position $b \in [0, 31]$:

1. A **predicate array** is computed:
   $$
   p_i =
   \begin{cases}
   1, & \text{if bit } b \text{ of } a_i = 0 \\
   0, & \text{otherwise}
   \end{cases}
   $$

2. An **exclusive prefix sum (scan)** over the predicate array computes output indices.

3. A **scatter operation** writes elements into the correct positions in the output buffer.

4. Input and output buffers are swapped for the next pass.

This approach guarantees **stable sorting**, which is a required property for Radix Sort correctness.

---

## 3. Handling Signed Integers

Radix Sort naturally operates on **unsigned integers**. To correctly sort **signed 32-bit integers**, the following technique is applied:

* During the **most significant bit (MSB) pass**, the sign bit is **inverted**.
* This transforms signed integer ordering into lexicographically sortable unsigned ordering.
* After the final pass, the resulting array reflects correct signed ordering.

This method is widely used in GPU radix sorting and avoids post-processing steps.

---

## 4. Project Structure

The project follows a **modular CUDA project layout**, separating kernels, benchmarks, and utilities.

```
Lab 4/
├── include/
│   ├── utils.h
│   └── cpu_sort.h
├── src/
│   ├── utils.cpp
│   ├── cpu_sort.cpp
│   ├── thrust_sort.cu
│   ├── radix_sort.cu
│   ├── benchmark.cu
│   └── main.cu
├── Makefile
└── README.md
```

### Structure Rationale

* `utils.h / utils.cpp`
  * Random data generation

* `cpu_sort.h / cpu_sort.cpp`
  * CPU reference sorting helpers

* `thrust_sort.cu`
  * GPU baseline using `thrust::sort`

* `radix_sort.cu`
  * Custom CUDA Radix Sort implementation

* `benchmark.cu`
  * Main execution
  * Timing
  * Correctness verification

* `main.cu`
  * Warm-ups
  * Main execution
---

## 5. Build System

The project is built using **CMake with CUDA support**.

### Key configuration details:

* C++17 and CUDA C++17
* Release build with `-O3`
* CUDA architecture targeting enabled
* Separate compilation for `.cu` files

The final executable:

* `radix_sort.exe`

---

## 6. CUDA Implementations

### 6.1 Thrust-Based GPU Baseline

The `thrust::sort` implementation serves as a **reference GPU baseline**.

Characteristics:

* Uses highly optimized internal algorithms
* Automatically selects radix-based or merge-based strategies
* Employs:

  * Multi-bit processing per pass
  * Shared memory
  * Warp-level primitives
* Minimal kernel launches

This implementation is used for both **performance comparison** and **correctness validation**.

---

### 6.2 Custom GPU Radix Sort Implementation

The custom implementation consists of the following GPU stages per bit:

1. **Predicate Kernel**

   * One thread per element
   * Extracts current bit
2. **Prefix-Sum (Scan)**

   * Implemented using `thrust::exclusive_scan`
3. **Scatter Kernel**

   * Computes final positions
   * Writes to output buffer
4. **Buffer Swap**

   * Alternates input/output arrays

All intermediate data structures (predicate array, scan array) reside in **global memory**.

---

## 7. Experimental Setup

* Data type: `int32`
* Input sizes:

  * $10^5$
  * $10^6$
  * $10^7$
  * $10^8$
* Data distribution:

  * Uniform random integers
  * Includes negative values
* Timing method:

  * `cudaEventElapsedTime` (GPU)
  * `std::chrono` (CPU)
* Build type: **Release**

Each measurement reflects the **average of multiple runs**.

---

## 8. Performance Results

### 8.1 Execution Time Comparison

|  Array Size | CPU std::sort (ms) | CPU qsort (ms) | GPU thrust::sort (ms) | GPU Radix Sort (ms) |
| ----------: | -----------------: | -------------: | --------------------: | ------------------: |
|     100,000 |               5.48 |           8.14 |              **0.17** |                1.96 |
|   1,000,000 |              65.44 |          95.01 |              **0.61** |               15.21 |
|  10,000,000 |             780.88 |        1095.73 |              **3.52** |               71.30 |
| 100,000,000 |            8850.89 |       15896.20 |             **69.34** |              626.36 |

---

## 9. Correctness Verification

Correctness is validated by comparing:

* Custom GPU Radix Sort output
* Against CPU `std::sort`
* And GPU `thrust::sort`

The final implementation **passes correctness checks for all tested input sizes**, including arrays with negative integers.

---

## 10. Analysis and Discussion

### 10.1 Kernel Launch and Scan Overhead

The custom Radix Sort performs:

* **32 kernel launches** for predicate computation
* **32 prefix-sum operations**
* **32 scatter kernel launches**

This results in **significant kernel launch and synchronization overhead**, which dominates execution time for moderate input sizes.

---

### 10.2 Memory Access Behavior

Key observations:

* All intermediate arrays are stored in **global memory**
* Each element is:

  * Read multiple times
  * Written multiple times per bit
* No shared-memory reuse
* No coalesced multi-bit processing

This leads to **high global memory traffic**, which limits performance.

---

### 10.3 Comparison with `thrust::sort`

`thrust::sort` significantly outperforms the naive implementation due to:

* Processing **multiple bits per pass**
* Reduced number of kernel launches
* Efficient shared-memory usage
* Warp-level primitives
* Architecture-specific optimizations

Although both algorithms have theoretical complexity $O(N)$, **constant factors dominate real-world GPU performance**.

---

## 11. Conclusion

In this laboratory work, a **fully functional GPU-based Radix Sort** was implemented using CUDA. The implementation:

* Correctly handles signed integers
* Demonstrates GPU acceleration over CPU-based sorting
* Provides a clear comparison against a highly optimized library solution

While the custom Radix Sort does not outperform `thrust::sort`, it successfully illustrates:

* The structure of scan-based parallel algorithms
* The performance impact of kernel launch overhead
* The importance of memory hierarchy optimization in CUDA

This work highlights a key practical insight:

> **Correct parallel algorithms are not necessarily efficient without careful consideration of GPU architecture and memory behavior.**

---