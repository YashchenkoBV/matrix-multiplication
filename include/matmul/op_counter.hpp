#pragma once
#include <cstdint>

namespace matmul {

// Counts "field" operations: one multiply/add at the scalar element type T level.
struct OpCounter {
  std::uint64_t mul = 0;
  std::uint64_t add = 0;

  void reset() noexcept { mul = 0; add = 0; }
};

} // namespace matmul
