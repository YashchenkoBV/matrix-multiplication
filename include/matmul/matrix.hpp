#pragma once
#include "matmul/aligned_alloc.hpp"
#include "matmul/config.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace matmul {

template <class T>
struct MatrixView {
  T* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t stride = 0; // elements between row starts

  T& operator()(std::size_t r, std::size_t c) {
#if MATMUL_ENABLE_CHECKS
    if (r >= rows || c >= cols) throw std::out_of_range("MatrixView index out of range");
#endif
    return data[r * stride + c];
  }
  const T& operator()(std::size_t r, std::size_t c) const {
#if MATMUL_ENABLE_CHECKS
    if (r >= rows || c >= cols) throw std::out_of_range("MatrixView index out of range");
#endif
    return data[r * stride + c];
  }

  MatrixView<T> subview(std::size_t r0, std::size_t c0, std::size_t rcount, std::size_t ccount) const {
#if MATMUL_ENABLE_CHECKS
    if (r0 + rcount > rows || c0 + ccount > cols) throw std::invalid_argument("subview out of range");
#endif
    return MatrixView<T>{data + r0 * stride + c0, rcount, ccount, stride};
  }
};

template <class T>
struct MatrixView<const T> {
  const T* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t stride = 0;

  const T& operator()(std::size_t r, std::size_t c) const {
#if MATMUL_ENABLE_CHECKS
    if (r >= rows || c >= cols) throw std::out_of_range("MatrixView index out of range");
#endif
    return data[r * stride + c];
  }

  MatrixView<const T> subview(std::size_t r0, std::size_t c0, std::size_t rcount, std::size_t ccount) const {
#if MATMUL_ENABLE_CHECKS
    if (r0 + rcount > rows || c0 + ccount > cols) throw std::invalid_argument("subview out of range");
#endif
    return MatrixView<const T>{data + r0 * stride + c0, rcount, ccount, stride};
  }
};

template <class T>
class Matrix {
 public:
  Matrix() = default;

  Matrix(std::size_t r, std::size_t c) { resize(r, c); }

  void resize(std::size_t r, std::size_t c) {
    rows_ = r;
    cols_ = c;
    stride_ = c; // contiguous row-major
    buf_.allocate(rows_ * stride_ * sizeof(T));
  }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t stride() const noexcept { return stride_; }

  T* data() noexcept { return static_cast<T*>(buf_.data()); }
  const T* data() const noexcept { return static_cast<const T*>(buf_.data()); }

  T& operator()(std::size_t r, std::size_t c) {
    return view()(r, c);
  }
  const T& operator()(std::size_t r, std::size_t c) const {
    return view()(r, c);
  }

  MatrixView<T> view() noexcept { return MatrixView<T>{data(), rows_, cols_, stride_}; }
  MatrixView<const T> view() const noexcept { return MatrixView<const T>{data(), rows_, cols_, stride_}; }

  void fill(const T& v) {
    auto* p = data();
    for (std::size_t i = 0; i < rows_ * stride_; ++i) p[i] = v;
  }

 private:
  std::size_t rows_ = 0, cols_ = 0, stride_ = 0;
  AlignedBuffer buf_{};
};

} // namespace matmul
