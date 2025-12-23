#pragma once
#include "matmul/matrix.hpp"
#include "matmul/op_counter.hpp"
#include "matmul/config.hpp"

#include <cstddef>
#include <stdexcept>

namespace matmul {

template <class T>
inline void gemm_naive_impl(const MatrixView<const T>& A,
                            const MatrixView<const T>& B,
                            MatrixView<T> C,
                            OpCounter* ops) {
#if MATMUL_ENABLE_CHECKS
  if (A.cols != B.rows) throw std::invalid_argument("gemm_naive: A.cols != B.rows");
  if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("gemm_naive: C dims mismatch");
#endif

  const std::size_t m = A.rows;
  const std::size_t k = A.cols;
  const std::size_t n = B.cols;

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (k == 0) { C(i, j) = T{}; continue; }

      T sum = A(i, 0) * B(0, j);
#if MATMUL_ENABLE_OPCOUNT
      if (ops) ops->mul += 1;
#endif
      for (std::size_t p = 1; p < k; ++p) {
        sum += A(i, p) * B(p, j);
#if MATMUL_ENABLE_OPCOUNT
        if (ops) { ops->mul += 1; ops->add += 1; } // sub counted as add elsewhere
#endif
      }
      C(i, j) = sum;
    }
  }
}

template <class T>
inline void gemm_naive(const MatrixView<const T>& A,
                       const MatrixView<const T>& B,
                       MatrixView<T> C,
                       OpCounter* ops = nullptr) {
  gemm_naive_impl<T>(A, B, C, ops);
}

template <class T>
inline void gemm_naive(const MatrixView<T>& A,
                       const MatrixView<T>& B,
                       MatrixView<T> C,
                       OpCounter* ops = nullptr) {
  MatrixView<const T> Ac{A.data, A.rows, A.cols, A.stride};
  MatrixView<const T> Bc{B.data, B.rows, B.cols, B.stride};
  gemm_naive_impl<T>(Ac, Bc, C, ops);
}

template <class T>
inline void gemm_naive(const MatrixView<const T>& A,
                       const MatrixView<T>& B,
                       MatrixView<T> C,
                       OpCounter* ops = nullptr) {
  MatrixView<const T> Bc{B.data, B.rows, B.cols, B.stride};
  gemm_naive_impl<T>(A, Bc, C, ops);
}

template <class T>
inline void gemm_naive(const MatrixView<T>& A,
                       const MatrixView<const T>& B,
                       MatrixView<T> C,
                       OpCounter* ops = nullptr) {
  MatrixView<const T> Ac{A.data, A.rows, A.cols, A.stride};
  gemm_naive_impl<T>(Ac, B, C, ops);
}

} // namespace matmul
