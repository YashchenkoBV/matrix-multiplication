#pragma once
#include "matmul/config.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace matmul {

struct TrackedAllocStats {
  std::size_t current_bytes = 0;
  std::size_t peak_bytes = 0;
};

inline std::atomic<std::size_t>& tracked_current_bytes() {
  static std::atomic<std::size_t> v{0};
  return v;
}
inline std::atomic<std::size_t>& tracked_peak_bytes() {
  static std::atomic<std::size_t> v{0};
  return v;
}

inline void tracked_add(std::size_t n) {
  auto cur = tracked_current_bytes().fetch_add(n, std::memory_order_relaxed) + n;
  auto pk = tracked_peak_bytes().load(std::memory_order_relaxed);
  while (cur > pk && !tracked_peak_bytes().compare_exchange_weak(pk, cur, std::memory_order_relaxed)) {}
}
inline void tracked_sub(std::size_t n) {
  tracked_current_bytes().fetch_sub(n, std::memory_order_relaxed);
}

inline TrackedAllocStats get_tracked_alloc_stats() {
  return {tracked_current_bytes().load(std::memory_order_relaxed),
          tracked_peak_bytes().load(std::memory_order_relaxed)};
}

class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t bytes) { allocate(bytes); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& o) noexcept { move_from(o); }
  AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
    if (this != &o) { release(); move_from(o); }
    return *this;
  }

  ~AlignedBuffer() { release(); }

  void allocate(std::size_t bytes) {
    release();
    if (bytes == 0) return;

    // Use aligned new (portable across MSVC/MinGW/Clang/GCC in C++17+)
    ptr_ = ::operator new(bytes, std::align_val_t(kAlignmentBytes));
    tracked_bytes_ = bytes; // exact requested size
    tracked_add(tracked_bytes_);
  }

  void release() noexcept {
    if (!ptr_) return;
    tracked_sub(tracked_bytes_);
    ::operator delete(ptr_, std::align_val_t(kAlignmentBytes));
    ptr_ = nullptr;
    tracked_bytes_ = 0;
  }

  void* data() noexcept { return ptr_; }
  const void* data() const noexcept { return ptr_; }
  std::size_t tracked_bytes() const noexcept { return tracked_bytes_; }

 private:
  void move_from(AlignedBuffer& o) noexcept {
    ptr_ = o.ptr_;
    tracked_bytes_ = o.tracked_bytes_;
    o.ptr_ = nullptr;
    o.tracked_bytes_ = 0;
  }

  void* ptr_ = nullptr;
  std::size_t tracked_bytes_ = 0;
};

} // namespace matmul
