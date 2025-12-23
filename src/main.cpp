#include "matmul/matrix.hpp"
#include "matmul/naive.hpp"
#include "matmul/op_counter.hpp"
#include "matmul/timer.hpp"
#include "matmul/memory_win.hpp"
#include "matmul/aligned_alloc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>   // FIX #1: needed for std::uint32_t
#include <iomanip>
#include <iostream>
#include <random>
#include <string> 

static void fill_random(matmul::Matrix<double>& M, std::uint32_t seed = 123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < M.rows(); ++i)
    for (std::size_t j = 0; j < M.cols(); ++j)
      M(i, j) = dist(rng);
}

static double max_abs_diff(const matmul::Matrix<double>& A, const matmul::Matrix<double>& B) {
  double md = 0.0;
  for (std::size_t i = 0; i < A.rows(); ++i)
    for (std::size_t j = 0; j < A.cols(); ++j)
      md = std::max(md, std::abs(A(i, j) - B(i, j)));
  return md;
}

int main(int argc, char** argv) {
  std::size_t n = 256;
  if (argc >= 2) n = static_cast<std::size_t>(std::stoul(argv[1]));

  matmul::Matrix<double> A(n, n), B(n, n), C(n, n);
  fill_random(A, 1);
  fill_random(B, 2);
  C.fill(0.0);

  // FIX #2: make const refs so view() returns MatrixView<const double>
  const matmul::Matrix<double>& Aconst = A;
  const matmul::Matrix<double>& Bconst = B;

  auto mem_before = matmul::get_process_memory_info();
  auto tracked_before = matmul::get_tracked_alloc_stats();
  (void)tracked_before; // not strictly needed, but kept for symmetry

  matmul::OpCounter ops{};
  matmul::Stopwatch sw;
  sw.start();
  matmul::gemm_naive<double>(Aconst.view(), Bconst.view(), C.view(), &ops);
  const double sec = sw.stop_seconds();

  auto mem_after = matmul::get_process_memory_info();
  auto tracked_after = matmul::get_tracked_alloc_stats();

  // Basic sanity check on a tiny case
  {
    matmul::Matrix<double> A2(2, 2), B2(2, 2), C2(2, 2), Cref(2, 2);
    A2(0,0)=1; A2(0,1)=2; A2(1,0)=3; A2(1,1)=4;
    B2(0,0)=5; B2(0,1)=6; B2(1,0)=7; B2(1,1)=8;

    const matmul::Matrix<double>& A2c = A2;
    const matmul::Matrix<double>& B2c = B2;

    // FIX #3: same const-view issue here
    matmul::gemm_naive<double>(A2c.view(), B2c.view(), C2.view(), nullptr);

    Cref(0,0)=1*5+2*7; Cref(0,1)=1*6+2*8;
    Cref(1,0)=3*5+4*7; Cref(1,1)=3*6+4*8;

    const double err = max_abs_diff(C2, Cref);
    if (err != 0.0) {
      std::cerr << "Sanity check failed (err=" << err << ")\n";
      return 1;
    }
  }

  const double gflops = (2.0 * static_cast<double>(n) * n * n) / (sec * 1e9);

  std::cout << "naive GEMM n=" << n << "\n";
  std::cout << "time: " << std::fixed << std::setprecision(6) << sec << " s\n";
  std::cout << "throughput: " << std::fixed << std::setprecision(3)
            << gflops << " GFLOP/s (2*n^3)\n";

  std::cout << "ops (field): mul=" << ops.mul << " add=" << ops.add << "\n";

  std::cout << "process memory (working set): "
            << mem_before.working_set_bytes / (1024.0*1024.0) << " MiB -> "
            << mem_after.working_set_bytes / (1024.0*1024.0) << " MiB\n";

  std::cout << "tracked allocations (ours): current="
            << tracked_after.current_bytes / (1024.0*1024.0) << " MiB, peak="
            << tracked_after.peak_bytes / (1024.0*1024.0) << " MiB\n";

  return 0;
}
