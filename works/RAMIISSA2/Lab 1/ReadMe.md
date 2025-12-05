# 🎨 CUDA ASCII Shape Renderer

This project demonstrates how to render simple geometric shapes (Square, Circle, Good Circle, Triangle) on the GPU using CUDA.
Each shape is drawn as ASCII art inside a fixed-size canvas, with pixel computation fully executed in CUDA kernels.

The project also includes a benchmarking utility (`test.cu`) for testing different block sizes and comparing GPU kernel performance.

---

## 📁 Project Structure

```
project/
│
├── include/
│   ├── circle.h
│   ├── good_circle.h
│   ├── square.h
│   └── triangle.h
│
├── src/
│   ├── circle.cu
│   ├── good_circle.cu
│   ├── square.cu
│   ├── triangle.cu
│   ├── main.cu
│   └── test.cu
│
├── shapes.exe
├── test.exe
└── README.md
```

---

## 🚀 How to Compile

### **Compile the main program**

Using NVCC from the *x64 Native Tools Command Prompt for VS 2022*:

```bash
nvcc src/main.cu src/circle.cu src/good_circle.cu src/square.cu src/triangle.cu -o shape.exe
```

### **Compile the benchmark tester**

```bash
nvcc src/test.cu src/circle.cu src/good_circle.cu src/square.cu src/triangle.cu -o test.exe
```

---

## ▶️ How to Run

### **Run the shape renderer**

```bash
.\shape.exe
```

Choose one of the options:

```
1. Square
2. Circle
3. Good Circle
4. Triangle
```

The ASCII art will be printed in the terminal.

### **Run the benchmark**

```bash
.\test.exe
```

This will test multiple block sizes and print a timing table.

---

## 🧪 Example Outputs

### **Square**

```
.........................................
.                                       .
.                                       .
.      ***************************      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      *                         *      .
.      ***************************      .
.                                       .
.                                       .
.                                       .
.........................................
```

### **Circle**

```
.........................................
.                                       .
.                 *****                 .
.               *       *               .
.              *         *              .
.             *           *             .
.            *             *            .
.                                       .
.           *               *           .
.           *               *           .
.           *               *           .
.           *               *           .
.           *               *           .
.                                       .
.            *             *            .
.             *           *             .
.              *         *              .
.               *       *               .
.                 *****                 .
.                                       .
.........................................
```

### **Good Circle**

```
.........................................
.                                       .
.              ***********              .
.          **               **          .
.        **                   **        .
.      **                       **      .
.    **                           **    .
.                                       .
.  **                               **  .
.  **                               **  .
.  **                               **  .
.  **                               **  .
.  **                               **  .
.                                       .
.    **                           **    .
.      **                       **      .
.        **                   **        .
.          **               **          .
.              ***********              .
.                                       .
.........................................
```

### **Triangle**

```
.........................................
.                                       .
.                                       .
.                   *                   .
.                  * *                  .
.                 *   *                 .
.                *     *                .
.               *       *               .
.              *         *              .
.             *           *             .
.            *             *            .
.           *               *           .
.          *                 *          .
.         *                   *         .
.        *                     *        .
.       *                       *       .
.      ***************************      .
.                                       .
.                                       .
.                                       .
.........................................
```

---

## 📊 Performance Results

These are GPU execution times averaged over multiple iterations using CUDA events.

|Block Size      | Square (ms)   | Circle (ms)   | Good Circle (ms)      | Triangle (ms)    |
| -------------: | ------------: | ------------: | --------------------: | ---------------: |
8x8              | 0.047         | 0.050         | 0.049                 | 0.046            |
16x16            | 0.035         | 0.023         | 0.024                 | 0.032            |
32x32            | 0.048         | 0.037         | 0.036                 | 0.045            |

---

## ⚙️ Implementation Details

* Each shape is rendered using a dedicated CUDA kernel.
* The canvas is a 1D `char` array of size `WIDTH × HEIGHT`.
* Block/grid dimensions can be tuned for performance.
* GPU timing is measured using CUDA events.
* The renderer uses ASCII characters:

  * `*` for shape borders
  * `.` for canvas edges
  * space `' '` for background

---

## 🧠 Conclusion

The performance tests show that **block size has a measurable impact** on execution time, even for lightweight ASCII-drawing kernels.

Among the tested configurations:

* **16×16 blocks consistently delivered the fastest execution** across all shapes.
* **8×8 and 32×32 blocks were slower**, likely due to:

  * Reduced occupancy (8×8 = too few threads per block)
  * Increased register pressure / scheduling inefficiency (32×32 = too many threads per block)
* The differences are small in absolute time (fractions of a millisecond), but the trend is stable across all kernels.

### ✔ Key Findings

* **16×16** appears to be the optimal block size for this workload.
* Shapes with more complex math (Circle, Good Circle) benefit more noticeably from proper block sizing.
* The workload is small (ASCII grid), so results mostly reflect **GPU scheduling overhead**, not heavy computation.

### 📌 Final Verdict

For this lab and similar grid-based CUDA tasks, **a 16×16 thread block configuration provides the best balance between parallelism and efficiency**.

---