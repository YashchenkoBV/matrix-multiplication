#pragma once
#include "matmul/matrix.hpp"
#include "matmul/op_counter.hpp"

#include <stdexcept>

namespace matmul {

struct AlphaEvolveConfig {
  // later: leaf/block size, padding rules, etc.
  bool pad_to_multiple_of_4 = true;
};

template <class T>
inline void gemm_alphaevolve_4x4(const MatrixView<const T>& A,
                                 const MatrixView<const T>& B,
                                 MatrixView<T> C,
                                 OpCounter* /*ops*/ = nullptr) {
  // Placeholder: will be implemented once Strassen + benchmarks are stable.
  throw std::logic_error("AlphaEvolve 4x4 not implemented yet");
}

template <class T>
inline void gemm_alphaevolve(const MatrixView<const T>& A,
                             const MatrixView<const T>& B,
                             MatrixView<T> C,
                             OpCounter* /*ops*/ = nullptr,
                             AlphaEvolveConfig /*cfg*/ = {}) {
  // Placeholder: will eventually do recursive blocking using the 4x4 kernel.
  throw std::logic_error("AlphaEvolve (blocked/recursive) not implemented yet");
}

} // namespace matmul
