#pragma once
#include <chrono>

namespace matmul {

class Stopwatch {
 public:
  using clock = std::chrono::steady_clock;

  void start() noexcept { t0_ = clock::now(); running_ = true; }
  double stop_seconds() noexcept {
    if (!running_) return 0.0;
    auto t1 = clock::now();
    running_ = false;
    return std::chrono::duration<double>(t1 - t0_).count();
  }

 private:
  clock::time_point t0_{};
  bool running_ = false;
};

} // namespace matmul
