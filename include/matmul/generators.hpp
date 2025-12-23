#pragma once
#include "matmul/matrix.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace matmul {

template <class T>
struct is_complex : std::false_type {};
template <class R>
struct is_complex<std::complex<R>> : std::true_type {};
template <class T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <class T>
struct scalar_of { using type = T; };
template <class R>
struct scalar_of<std::complex<R>> { using type = R; };
template <class T>
using scalar_of_t = typename scalar_of<T>::type;

enum class MatrixKind {
  Zeros,
  Identity,           // square only
  RandomUniform,      // iid uniform (real or complex)
  SymmetricUniform,   // square only: A = A^T (for complex, exact transpose mirror)
  HermitianUniform    // square only (complex recommended): A = A^H (conjugate transpose)
};

template <class T>
inline void fill_matrix(MatrixView<T> M,
                        MatrixKind kind,
                        std::uint64_t seed = 123,
                        scalar_of_t<T> lo = static_cast<scalar_of_t<T>>(-1),
                        scalar_of_t<T> hi = static_cast<scalar_of_t<T>>(1)) {
  using S = scalar_of_t<T>;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<S> dist(lo, hi);

  auto zero_all = [&]() {
    for (std::size_t i = 0; i < M.rows; ++i)
      for (std::size_t j = 0; j < M.cols; ++j)
        M(i, j) = T{};
  };

  if (kind == MatrixKind::Zeros) {
    zero_all();
    return;
  }

  if (kind == MatrixKind::Identity) {
    if (M.rows != M.cols) throw std::invalid_argument("Identity requires square matrix");
    zero_all();
    for (std::size_t i = 0; i < M.rows; ++i) M(i, i) = T{1};
    return;
  }

  if (kind == MatrixKind::RandomUniform) {
    for (std::size_t i = 0; i < M.rows; ++i) {
      for (std::size_t j = 0; j < M.cols; ++j) {
        if constexpr (is_complex_v<T>) {
          M(i, j) = T{dist(rng), dist(rng)};
        } else {
          M(i, j) = dist(rng);
        }
      }
    }
    return;
  }

  if (kind == MatrixKind::SymmetricUniform) {
    if (M.rows != M.cols) throw std::invalid_argument("SymmetricUniform requires square matrix");
    const std::size_t n = M.rows;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i; j < n; ++j) {
        T v;
        if constexpr (is_complex_v<T>) v = T{dist(rng), dist(rng)};
        else v = dist(rng);
        M(i, j) = v;
        M(j, i) = v; // exact transpose symmetry (no conjugation)
      }
    }
    return;
  }

  if (kind == MatrixKind::HermitianUniform) {
    if (M.rows != M.cols) throw std::invalid_argument("HermitianUniform requires square matrix");
    const std::size_t n = M.rows;

    if constexpr (!is_complex_v<T>) {
      // For real types, Hermitian == Symmetric
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
          T v = dist(rng);
          M(i, j) = v;
          M(j, i) = v;
        }
      }
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        // diagonal must be real for Hermitian
        M(i, i) = T{dist(rng), S{0}};
        for (std::size_t j = i + 1; j < n; ++j) {
          T v{dist(rng), dist(rng)};
          M(i, j) = v;
          M(j, i) = std::conj(v);
        }
      }
    }
    return;
  }

  throw std::invalid_argument("fill_matrix: unknown MatrixKind");
}

// Convenience for owning Matrix<T>
template <class T>
inline Matrix<T> make_matrix(std::size_t rows,
                             std::size_t cols,
                             MatrixKind kind,
                             std::uint64_t seed = 123,
                             scalar_of_t<T> lo = static_cast<scalar_of_t<T>>(-1),
                             scalar_of_t<T> hi = static_cast<scalar_of_t<T>>(1)) {
  Matrix<T> M(rows, cols);
  fill_matrix<T>(M.view(), kind, seed, lo, hi);
  return M;
}

} // namespace matmul
