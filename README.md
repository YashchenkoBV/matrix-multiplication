# Matrix Multiplication Report

**Course:** Algorithm Engineering  
**Student:** Bogdan Iashchenko

## Goal

The goal of this project is to implement matrix-multiplication algorithms in C++, benchmark them across matrix sizes and matrix types, and analyze their performance.

## Algorithms implemented

### Naive GEMM (baseline)
Computes \(C = A \cdot B\) using the straightforward triple-loop algorithm:
- Multiplications: \(n^3\)
- Additions: \(n^2(n-1)\)

This method has a lot of arithmetic work but, in the same time, highest simplicity and predictable memory access.

### Strassen (classic)
Strassen’s method recursively splits matrices into quadrants and computes the result using **7** recursive multiplications (instead of 8), plus additional additions/subtractions.

For power-of-two sizes \(n=2^k\), pure Strassen (down to 1×1) has:
- Multiplications: \(7^k = 7^{\log_2 n}\)  
- Additions: grows as \(O(n^{\log_2 7})\)

### Data types
Benchmarks were run for:
- **Real:** `double`
- **Complex:** `std::complex<double>`

For complex arithmetic, I also report equivalent number of operations for real numbers for interpretation:
- complex add ≈ 2 real adds  
- complex mul ≈ 4 real muls + 2 real adds

### Matrix types
Matrices were generated with reproducible PRNG seeds per size, using:
- **random:** i.i.d uniform entries in \([-1,1]\)
- **symmetric:** \(A = A^T\)
- **hermitian (complex only):** \(A = A^H\) (conjugate transpose)

---

## Benchmark methodology

### Sizes
Benchmarked square matrices with sizes:
\[
n \in \{2,4,8,16,32,64,128,256,512\}
\]

## Benchmark results 

`naive/Strassen` > 1 would mean Strassen is faster. In my results, Strassen is slower for all tested sizes up to 512, but the gap shrinks for complex data.

### 5.1 Real, random

| n | naive (ms) | Strassen (ms) | naive/Strassen | mul reduction × | ΔWS naive (MiB) | ΔWS Strassen (MiB) | scratch est. (MiB) |
|---|---|---|---|---|---|---|---|
| 64 | 0.142090 | 5.1341 | 0.027676 | 2.2282 | 0.023438 | 0.117188 | 0.095551 |
| 128 | 1.4232 | 23.133 | 0.061523 | 2.5465 | 0.121094 | 0.492188 | 0.376938 |
| 256 | 13.841 | 151.5 | 0.091336 | 2.9103 | 0.492188 | 1.9961 | 1.5021 |
| 512 | 233.9 | 820.1 | 0.285181 | 3.3260 | 2.1250 | 8.0039 | 6.0022 |

**Interpretation:** Strassen reduces multiplications (3.33× fewer at n=512), but is still ~3.5× slower in wall-time at n=512 in this implementation.

### 5.2 Real, symmetric

| n | naive (ms) | Strassen (ms) | naive/Strassen | mul reduction × | ΔWS naive (MiB) | ΔWS Strassen (MiB) | scratch est. (MiB) |
|---|---|---|---|---|---|---|---|
| 64 | 0.293300 | 4.3799 | 0.066966 | 2.2282 | 0.023438 | 0.117188 | 0.095551 |
| 128 | 2.7260 | 18.554 | 0.146904 | 2.5465 | 0.121094 | 0.492188 | 0.376938 |
| 256 | 22.868 | 143.3 | 0.159526 | 2.9103 | 0.492188 | 1.9961 | 1.5021 |
| 512 | 343.4 | 720.7 | 0.476457 | 3.3260 | 2.1250 | 8.0039 | 6.0022 |

**Interpretation:** symmetric inputs are slightly slower overall here (likely cache/memory effects), and Strassen is still slower, but closer.

### 5.3 Complex, random

| n | naive (ms) | Strassen (ms) | naive/Strassen | mul reduction × | ΔWS naive (MiB) | ΔWS Strassen (MiB) | scratch est. (MiB) |
|---|---|---|---|---|---|---|---|
| 64 | 0.304400 | 1.8695 | 0.162552 | 2.2282 | 0.054688 | 0.246094 | 0.189335 |
| 128 | 2.4677 | 15.937 | 0.154856 | 2.5465 | 0.246094 | 0.996094 | 0.752015 |
| 256 | 21.858 | 105.4 | 0.207297 | 2.9103 | 1.0000 | 4.0039 | 3.0016 |
| 512 | 348.9 | 637.1 | 0.547711 | 3.3260 | 4.0000 | 16.000 | 12.003 |

**Interpretation:** Strassen is still slower, but much closer for complex data (only ~1.8× slower at n=512). This is consistent with multiplications being relatively more expensive in complex arithmetic.

### 5.4 Complex, symmetric

| n | naive (ms) | Strassen (ms) | naive/Strassen | mul reduction × | ΔWS naive (MiB) | ΔWS Strassen (MiB) | scratch est. (MiB) |
|---|---|---|---|---|---|---|---|
| 64 | 0.388600 | 1.7994 | 0.215952 | 2.2282 | 0.054688 | 0.246094 | 0.189335 |
| 128 | 3.2524 | 15.014 | 0.216589 | 2.5465 | 0.246094 | 0.996094 | 0.752015 |
| 256 | 30.379 | 114.1 | 0.266286 | 2.9103 | 1.0000 | 4.0039 | 3.0016 |
| 512 | 433.3 | 627.4 | 0.690682 | 3.3260 | 4.0000 | 16.000 | 12.003 |

**Interpretation:** This is the closest case; Strassen is only ~1.45× slower at n=512.


## Analysis

## Operation counts

### Theory
- **Naive (n×n):**
  - multiplications: \(n^3\)
  - additions: \(n^2(n-1)\)

- **Strassen (pure, n=2^k, leaf=1):**
  - multiplications: \(7^k = 7^{\log_2 n}\)
  - additions: increases due to:
    - 10 matrix additions/subtractions to form Strassen’s intermediate sums/differences (per level)
    - plus combination additions to assemble C blocks

### Confirming with observed counts
At \(n=512\) (real, random):
- Naive:
  - mul = 134,217,728
  - add = 133,955,584
  - total field ops = 268,173,312
- Strassen (leaf=1):
  - mul = 40,353,607 (**3.326× fewer muls**)
  - add = 240,548,778 (**more adds than naive**)
  - total field ops = 280,902,385 (**slightly higher than naive**)

This explains a key outcome: **Strassen reduces multiplications but increases additions so much that total scalar ops are not necessarily lower for real arithmetic at these sizes.**

For **complex arithmetic**, real-equivalent ops at \(n=512\) (random) show:
- Naive real-equivalent ops ≈ 1,073,217,536
- Strassen real-equivalent ops ≈ 723,219,198  
So Strassen provides a clearer theoretical arithmetic advantage in complex arithmetic, which matches the smaller runtime gap in complex benchmarks.

---

## Memory overhead and access patterns

### What Strassen needs
Strassen requires many temporary matrices per recursion level (sums/differences + subproducts). My implementation uses a preallocated **scratch arena**, so allocations are avoided during timed trials, but the scratch still increases memory footprint and memory traffic.

### What the benchmarks show
For real random at \(n=512\):
- Naive: Δ working set ≈ **2.13 MiB**
- Strassen: Δ working set ≈ **8.00 MiB**
- Strassen scratch estimate: **~6.00 MiB**

For complex random at \(n=512\):
- Naive: Δ working set ≈ **4.00 MiB**
- Strassen: Δ working set ≈ **16.00 MiB**
- Strassen scratch estimate: **~12.00 MiB**

**Interpretation:** Strassen’s extra temporaries increase memory footprint and cause substantially more data movement. On modern CPUs, memory traffic and cache behavior can dominate runtime, especially when the algorithm is not carefully blocked/optimized. This is a likely contributor to Strassen being slower up to n=512 in this implementation.

---

## Numerical stability

Naive GEMM tends to be more numerically stable because it performs the “direct” sum of products.  
Strassen introduces many extra additions/subtractions of intermediate matrices, which can amplify rounding error via cancellation (especially when values have mixed signs and similar magnitudes).

In this submission I did not compute explicit error norms in the CSV output. However, based on numerical-analysis expectations:
- **Strassen is expected to introduce more floating-point error** than naive, particularly at larger depths (smaller leaf) and larger n.

A straightforward validation (recommended if time permits) is:
- compute `max_abs_diff` or relative Frobenius error between naive and Strassen outputs for the same inputs (for several sizes and both real/complex), and report how it grows with n.

---

## Hardware and parallelism impact

These measurements were performed on a **single CPU thread** on Windows using a Release/`-O3` build (Ninja). Practical high-performance GEMM libraries typically outperform both of these implementations because they use:
- cache blocking / tiling,
- SIMD vectorization,
- multithreading,
- and architecture-specific microkernels.

Strassen provides an asymptotic multiplication reduction, but on CPUs:
- the extra temporaries and additions increase memory bandwidth needs,
- recursion overhead increases,
- and without blocking, cache locality is poor.

Parallelism considerations:
- Strassen has **7 independent subproducts** per recursion level, which is conceptually parallelizable.
- In practice, scaling depends on managing scratch space and memory bandwidth; naive blocked GEMM is often easier to optimize and parallelize effectively.

Given my results (up to n=512), the “theoretical advantage” of Strassen did **not** translate into a speedup on this hardware/software setup.

---

## Overall takeaways

1. **Strassen achieved the expected multiplication-count reduction** (e.g., 3.33× fewer multiplications at n=512), confirming the algorithmic theory.
2. **Wall-clock performance did not improve** up to n=512 in this implementation; naive was faster in all tested scenarios.
3. The primary reasons are consistent with algorithm-engineering trade-offs:
   - **more additions** (for real, total field ops were slightly higher than naive at n=512),
   - **more temporary memory and traffic** (working-set delta and scratch were much larger),
   - recursion overhead and lack of cache blocking.
4. Strassen became **closer** to naive in the complex benchmarks because multiplication is relatively more expensive there; real-equivalent op estimates suggest Strassen reduces total real-equivalent arithmetic for complex.
5. Overall, these results illustrate a core lesson: **asymptotic improvements do not guarantee practical speedups** at moderate sizes, especially without careful memory- and hardware-aware optimization.

---

## How to run

Built with Ninja:
```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first
.\build\matmul_bench.exe --algo naive    --dtype real    --kind random    --max 512 --warmups 2 --trials 10 --out naive_real_random.csv
```