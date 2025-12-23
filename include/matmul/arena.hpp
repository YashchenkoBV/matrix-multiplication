#pragma once
#include "matmul/aligned_alloc.hpp"
#include "matmul/config.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace matmul {

// Scratch allocations do NOT need 64B alignment each time.
// Use 16B (SIMD-friendly) or alignof(T), whichever is larger.
inline constexpr std::size_t kScratchAlignmentBytes = 16;

template <class T>
class ScratchArena {
 public:
  ScratchArena() = default;
  explicit ScratchArena(std::size_t capacity_bytes) { reset_capacity(capacity_bytes); }

  void reset_capacity(std::size_t capacity_bytes) {
    buf_.allocate(capacity_bytes);
    cap_bytes_ = buf_.tracked_bytes();
    off_bytes_ = 0;
  }

  std::size_t capacity_bytes() const noexcept { return cap_bytes_; }
  std::size_t used_bytes() const noexcept { return off_bytes_; }

  std::size_t mark() const noexcept { return off_bytes_; }

  void rollback(std::size_t m) {
    if (m > off_bytes_) throw std::invalid_argument("ScratchArena::rollback: mark out of range");
    off_bytes_ = m;
  }

  MatrixView<T> alloc_matrix(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) return MatrixView<T>{nullptr, rows, cols, cols};

    const std::size_t bytes = rows * cols * sizeof(T);

    const std::size_t align = std::max<std::size_t>(kScratchAlignmentBytes, alignof(T));
    std::size_t aligned_off = align_up(off_bytes_, align);

    if (aligned_off + bytes > cap_bytes_) {
      throw std::bad_alloc();
    }

    auto* base = static_cast<std::byte*>(buf_.data());
    T* ptr = reinterpret_cast<T*>(base + aligned_off);

    off_bytes_ = aligned_off + bytes;
    return MatrixView<T>{ptr, rows, cols, cols}; // contiguous row-major
  }

 private:
  static std::size_t align_up(std::size_t x, std::size_t a) {
    return (x + (a - 1)) & ~(a - 1);
  }

  AlignedBuffer buf_{};
  std::size_t cap_bytes_ = 0;
  std::size_t off_bytes_ = 0;
};

} // namespace matmul
