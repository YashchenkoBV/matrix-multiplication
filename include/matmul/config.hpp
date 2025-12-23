#pragma once
#include <cstddef>

#ifndef MATMUL_ENABLE_CHECKS
#define MATMUL_ENABLE_CHECKS 1
#endif

#ifndef MATMUL_ENABLE_OPCOUNT
#define MATMUL_ENABLE_OPCOUNT 1
#endif

namespace matmul {
inline constexpr std::size_t kAlignmentBytes = 64; // cacheline-friendly alignment
} // namespace matmul
