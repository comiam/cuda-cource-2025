# CUDA Laboratory Work №3 (Sobel Operator)

----------

## 1. Task Description

The objective of this laboratory work is to implement the **Sobel edge detection operator** using **CUDA** for GPU acceleration. The program loads a grayscale image, applies the Sobel filter on the GPU, and saves the resulting edge map to a file. Two GPU implementations are developed and compared:

1.  A **naive global-memory-based Sobel kernel**
    
2.  An **optimized shared-memory Sobel kernel** using tiling
    

----------

## 2. Sobel Operator

The Sobel operator computes image gradients in the horizontal and vertical directions using the following convolution kernels:

**Horizontal gradient (Gx):**

```
-1  0  1
-2  0  2
-1  0  1

```

**Vertical gradient (Gy):**

```
-1 -2 -1
 0  0  0
 1  2  1

```

The gradient magnitude is computed as:

$$  
\text{Magnitude} = \sqrt{G_x^2 + G_y^2}  
$$

The result is clamped to the range $[0, 255]$.

----------

## 3. Project Structure

```
Lab 3/
├── CMakeLists.txt
├── include/
│   └── image_io.h
├── src/
│   ├── image_io.cpp
│   ├── sobel.cu
│   └── sobel_shared.cu
├── assets/
│   └── pgm samples

```

----------

## 4. Build System

The project is built using **CMake** with CUDA support.

### Key configuration details:

-   C++17 and CUDA C++17
    
-   Release optimization enabled (`-O3`)
    
-   Line information enabled for profiling (`-lineinfo`)
    
-   Native GPU architecture targeting
    

Two executables are produced:

-   `sobel_gpu` — naive global-memory implementation
    
-   `sobel_gpu_shared` — shared-memory optimized implementation
    

----------

## 5. Image I/O

The program supports the **PGM (P5)** grayscale image format, which satisfies the minimum lab requirements.

Implemented functionality:

-   Loading binary PGM images
    
-   Skipping comment lines
    
-   Saving processed images back to PGM format
    

----------

## 6. CUDA Implementations

### 6.1 Naive Global Memory Kernel

-   Each thread processes one pixel
    
-   All neighboring pixels are read directly from global memory
    
-   No reuse of pixel data between threads
    
-   Simple control flow and minimal instruction count
    

This version serves as a **baseline implementation**.

----------

### 6.2 Shared Memory Kernel

-   Uses **16×16 thread blocks**
    
-   Loads a tile of pixels into shared memory
    
-   Includes a **1-pixel halo** on all sides
    
-   Synchronizes threads before computation
    
-   Each pixel is read from global memory only once per block
    

This implementation reduces redundant global memory accesses at the cost of:

-   Additional instructions
    
-   Boundary checks
    
-   Synchronization overhead
    

----------

## 7. Experimental Setup

-   Image resolution: **3840 × 2160**
    
-   Measurement method: `cudaEventElapsedTime`
    
-   Profiling tool: **NVIDIA Nsight Compute**
    
-   Build type: **Release**
    

----------

## 8. Performance Results

### 8.1 Kernel Execution Time Example for (3840 x 2160) image

| Kernel        | Time (ms)    | 
| ------------- | ------------ |
| Global memory | **0.320 ms** |
| Shared memory | **0.588 ms** |

----------

### 8.2 Nsight Compute Profiling Metrics

| Metric                            | Global     | Shared        |
| --------------------------------- | ---------- | ------------- |
| Excessive L2 sectors              | 3,869,292  | **1,034,160** |
| Avg bytes per global load sector  | 11.66      | 6.01          |
| Avg bytes per global store sector | 15.99      | 15.99         |
| L1TEX throughput (% peak)         | 67.48%     | 54.71%        |
| Instructions executed (avg)       | 243,137    | **456,116**   |

----------

## 9. Analysis and Discussion

### 9.1 Global Memory Traffic

The shared-memory implementation reduces excessive global memory transactions by approximately:

$$ 
\frac{3{,}869{,}292}{1{,}034{,}160} \approx 3.7\times  
$$

This confirms that **shared-memory tiling significantly reduces redundant global loads**, which is the primary goal of the optimization.

----------

### 9.2 Load and Store Efficiency

The average number of bytes utilized per global load sector is lower in the shared-memory kernel. This behavior is expected because:

-   Most global loads are eliminated
    
-   Remaining loads correspond to **halo regions**
    
-   Halo accesses are inherently uncoalesced in stencil algorithms
    

Global store efficiency is identical in both kernels and is limited by the **8-bit output format**, making further optimization impractical without changing the data representation.

----------

### 9.3 Instruction Count and Execution Time

The shared-memory kernel executes nearly **1.9× more instructions**, due to:

-   Explicit halo loading
    
-   Boundary condition handling
    
-   Synchronization (`__syncthreads()`)
    

As a result, despite improved memory behavior, the shared-memory kernel exhibits **higher execution time** for this image size on modern GPU hardware.

----------

## 10. Conclusion

In this laboratory work, two CUDA-based Sobel filter implementations were developed and analyzed. While the shared-memory version does not outperform the naive implementation in terms of execution time, profiling results clearly demonstrate a **significant reduction in global memory traffic**.

This experiment highlights an important practical insight:

> **Reducing memory traffic does not always translate to lower execution time**, especially when additional instruction overhead and synchronization are introduced.

Both implementations are correct, and the shared-memory kernel successfully demonstrates the principles of **memory hierarchy optimization** in CUDA.

----------