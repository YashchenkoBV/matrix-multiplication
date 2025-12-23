#pragma once
#include "matmul/arena.hpp"
#include "matmul/matrix.hpp"
#include "matmul/naive.hpp"
#include "matmul/op_counter.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace matmul {

struct StrassenConfig {
  std::size_t leaf_size = 1;          
  bool pad_to_power_of_two = true;   
};

inline bool is_power_of_two(std::size_t n) { return n && ((n & (n - 1)) == 0); }

inline std::size_t next_power_of_two(std::size_t n) {
  if (n == 0) return 1;
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

// Safe scratch cap for this implementation (see recurrence: peak < 3*n^2 elements)
template <class T>
inline std::size_t strassen_scratch_bytes(std::size_t n) {
  // Peak data (without alignment waste) is < 3*n^2 elements for this implementation.
  // Add a small alignment overhead term: <= 9 allocations per level.
  if (n == 0) return 0;

  std::uint64_t nn = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
  std::uint64_t data_bytes = 3ull * nn * static_cast<std::uint64_t>(sizeof(T));

  // levels = log2(n) for power-of-two n
  std::uint64_t levels = 0;
  std::size_t t = n;
  while (t > 1) { t >>= 1; ++levels; }

  std::uint64_t align = matmul::kScratchAlignmentBytes;
  std::uint64_t overhead = 9ull * levels * align + 1024ull; // tiny safety margin

  std::uint64_t total = data_bytes + overhead;

  if (total > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("strassen_scratch_bytes overflow");

  return static_cast<std::size_t>(total);
}


template <class T>
inline void mat_add(const MatrixView<const T>& A, const MatrixView<const T>& B, MatrixView<T> Out, OpCounter* ops) {
#if MATMUL_ENABLE_CHECKS
  if (A.rows != B.rows || A.cols != B.cols || Out.rows != A.rows || Out.cols != A.cols)
    throw std::invalid_argument("mat_add: dimension mismatch");
#endif
  for (std::size_t i = 0; i < A.rows; ++i)
    for (std::size_t j = 0; j < A.cols; ++j) {
      Out(i, j) = A(i, j) + B(i, j);
#if MATMUL_ENABLE_OPCOUNT
      if (ops) ops->add += 1;
#endif
    }
}

template <class T>
inline void mat_sub(const MatrixView<const T>& A, const MatrixView<const T>& B, MatrixView<T> Out, OpCounter* ops) {
#if MATMUL_ENABLE_CHECKS
  if (A.rows != B.rows || A.cols != B.cols || Out.rows != A.rows || Out.cols != A.cols)
    throw std::invalid_argument("mat_sub: dimension mismatch");
#endif
  for (std::size_t i = 0; i < A.rows; ++i)
    for (std::size_t j = 0; j < A.cols; ++j) {
      Out(i, j) = A(i, j) - B(i, j);
#if MATMUL_ENABLE_OPCOUNT
      if (ops) ops->add += 1; // count subtraction as one add-op
#endif
    }
}

namespace detail {

template <class T>
inline void strassen_rec(const MatrixView<const T>& A,
                         const MatrixView<const T>& B,
                         MatrixView<T> C,
                         ScratchArena<T>& arena,
                         OpCounter* ops,
                         std::size_t leaf) {
#if MATMUL_ENABLE_CHECKS
  if (A.rows != A.cols || B.rows != B.cols || C.rows != C.cols) throw std::invalid_argument("strassen_rec: non-square");
  if (A.rows != B.rows || A.rows != C.rows) throw std::invalid_argument("strassen_rec: size mismatch");
#endif

  const std::size_t n = A.rows;
  if (n == 0) return;

  if (n <= leaf) { gemm_naive<T>(A, B, C, ops); return; }
  if (n % 2 != 0) throw std::invalid_argument("strassen_rec: n must be even (pad first)");

  const auto base = arena.mark();
  const std::size_t m = n / 2;

  auto A11 = A.subview(0, 0, m, m);
  auto A12 = A.subview(0, m, m, m);
  auto A21 = A.subview(m, 0, m, m);
  auto A22 = A.subview(m, m, m, m);

  auto B11 = B.subview(0, 0, m, m);
  auto B12 = B.subview(0, m, m, m);
  auto B21 = B.subview(m, 0, m, m);
  auto B22 = B.subview(m, m, m, m);

  auto C11 = C.subview(0, 0, m, m);
  auto C12 = C.subview(0, m, m, m);
  auto C21 = C.subview(m, 0, m, m);
  auto C22 = C.subview(m, m, m, m);

  auto M1 = arena.alloc_matrix(m, m);
  auto M2 = arena.alloc_matrix(m, m);
  auto M3 = arena.alloc_matrix(m, m);
  auto M4 = arena.alloc_matrix(m, m);
  auto M5 = arena.alloc_matrix(m, m);
  auto M6 = arena.alloc_matrix(m, m);
  auto M7 = arena.alloc_matrix(m, m);

  auto T1 = arena.alloc_matrix(m, m);
  auto T2 = arena.alloc_matrix(m, m);

  mat_add(A11, A22, T1, ops);  mat_add(B11, B22, T2, ops);
  strassen_rec<T>({T1.data, m, m, T1.stride}, {T2.data, m, m, T2.stride}, M1, arena, ops, leaf);

  mat_add(A21, A22, T1, ops);
  strassen_rec<T>({T1.data, m, m, T1.stride}, B11, M2, arena, ops, leaf);

  mat_sub(B12, B22, T2, ops);
  strassen_rec<T>(A11, {T2.data, m, m, T2.stride}, M3, arena, ops, leaf);

  mat_sub(B21, B11, T2, ops);
  strassen_rec<T>(A22, {T2.data, m, m, T2.stride}, M4, arena, ops, leaf);

  mat_add(A11, A12, T1, ops);
  strassen_rec<T>({T1.data, m, m, T1.stride}, B22, M5, arena, ops, leaf);

  mat_sub(A21, A11, T1, ops);  mat_add(B11, B12, T2, ops);
  strassen_rec<T>({T1.data, m, m, T1.stride}, {T2.data, m, m, T2.stride}, M6, arena, ops, leaf);

  mat_sub(A12, A22, T1, ops);  mat_add(B21, B22, T2, ops);
  strassen_rec<T>({T1.data, m, m, T1.stride}, {T2.data, m, m, T2.stride}, M7, arena, ops, leaf);

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      // C11 = M1 + M4 - M5 + M7  (3 adds)
      T v11 = M1(i,j) + M4(i,j);
      v11 = v11 - M5(i,j);
      v11 = v11 + M7(i,j);

      // C12 = M3 + M5 (1)
      T v12 = M3(i,j) + M5(i,j);

      // C21 = M2 + M4 (1)
      T v21 = M2(i,j) + M4(i,j);

      // C22 = M1 - M2 + M3 + M6 (3)
      T v22 = M1(i,j) - M2(i,j);
      v22 = v22 + M3(i,j);
      v22 = v22 + M6(i,j);

#if MATMUL_ENABLE_OPCOUNT
      if (ops) ops->add += 8; // 3+1+1+3
#endif
      C11(i,j) = v11;
      C12(i,j) = v12;
      C21(i,j) = v21;
      C22(i,j) = v22;
    }
  }

  arena.rollback(base);
}

} // namespace detail

// Preallocated scratch version: for power-of-two square matrices only (benchmark-friendly).
template <class T>
inline void gemm_strassen_pow2_prealloc(const MatrixView<const T>& A,
                                        const MatrixView<const T>& B,
                                        MatrixView<T> C,
                                        ScratchArena<T>& arena,
                                        OpCounter* ops = nullptr,
                                        StrassenConfig cfg = {}) {
#if MATMUL_ENABLE_CHECKS
  if (A.cols != B.rows) throw std::invalid_argument("gemm_strassen: A.cols != B.rows");
  if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("gemm_strassen: C dims mismatch");
  if (A.rows != A.cols || B.rows != B.cols || C.rows != C.cols) throw std::invalid_argument("gemm_strassen: square only");
  if (A.rows != B.rows || A.rows != C.rows) throw std::invalid_argument("gemm_strassen: size mismatch");
  if (!is_power_of_two(A.rows)) throw std::invalid_argument("gemm_strassen_pow2_prealloc: n must be power of two");
#endif
  detail::strassen_rec<T>(A, B, C, arena, ops, cfg.leaf_size);
}

// Convenience wrapper (allocates scratch internally; fine for correctness, less ideal for benchmarks).
template <class T>
inline void gemm_strassen(const MatrixView<const T>& A,
                          const MatrixView<const T>& B,
                          MatrixView<T> C,
                          OpCounter* ops = nullptr,
                          StrassenConfig cfg = {}) {
#if MATMUL_ENABLE_CHECKS
  if (A.rows != A.cols || B.rows != B.cols || C.rows != C.cols)
    throw std::invalid_argument("gemm_strassen: square only");
#endif
  const std::size_t n = A.rows;
  if (n == 0) return;

  if (!cfg.pad_to_power_of_two) {
    if (!is_power_of_two(n)) throw std::invalid_argument("gemm_strassen: n not power-of-two and padding disabled");
    ScratchArena<T> arena(strassen_scratch_bytes<T>(n));
    detail::strassen_rec<T>(A, B, C, arena, ops, cfg.leaf_size);
    return;
  }

  const std::size_t np = is_power_of_two(n) ? n : next_power_of_two(n);
  if (np == n) {
    ScratchArena<T> arena(strassen_scratch_bytes<T>(n));
    detail::strassen_rec<T>(A, B, C, arena, ops, cfg.leaf_size);
    return;
  }

  Matrix<T> Ap(np, np), Bp(np, np), Cp(np, np);
  Ap.fill(T{}); Bp.fill(T{}); Cp.fill(T{});
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) { Ap(i,j) = A(i,j); Bp(i,j) = B(i,j); }

  ScratchArena<T> arena(strassen_scratch_bytes<T>(np));
  detail::strassen_rec<T>(Ap.view(), Bp.view(), Cp.view(), arena, ops, cfg.leaf_size);

  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) C(i,j) = Cp(i,j);
}

} // namespace matmul
