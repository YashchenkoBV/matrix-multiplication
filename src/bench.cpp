#include "matmul/generators.hpp"
#include "matmul/memory_win.hpp"
#include "matmul/naive.hpp"
#include "matmul/strassen.hpp"
#include "matmul/timer.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static volatile double g_sink = 0.0;

struct Args {
  std::string algo = "naive";      // naive | strassen
  std::string dtype = "real";      // real | complex
  std::string kind  = "random";    // random | symmetric | hermitian
  std::size_t max_n = 512;
  int warmups = 2;
  int trials  = 10;
  std::size_t leaf = 64;           // Strassen leaf
  std::string out = "results.csv";
};

static std::size_t parse_size_t(const char* s) {
  return static_cast<std::size_t>(std::stoull(std::string(s)));
}
static int parse_int(const char* s) {
  return std::stoi(std::string(s));
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { throw std::runtime_error(std::string("Missing value for ") + name); }
      return argv[++i];
    };

    if (k == "--algo") a.algo = need("--algo");
    else if (k == "--dtype") a.dtype = need("--dtype");
    else if (k == "--kind") a.kind = need("--kind");
    else if (k == "--max") a.max_n = parse_size_t(need("--max"));
    else if (k == "--warmups") a.warmups = parse_int(need("--warmups"));
    else if (k == "--trials") a.trials = parse_int(need("--trials"));
    else if (k == "--leaf") a.leaf = parse_size_t(need("--leaf"));
    else if (k == "--out") a.out = need("--out");
    else if (k == "--help") {
      std::cout <<
        "Usage: matmul_bench [options]\n"
        "  --algo   naive|strassen\n"
        "  --dtype  real|complex\n"
        "  --kind   random|symmetric|hermitian\n"
        "  --max    512 (max power-of-two size, starting from 2)\n"
        "  --warmups 2\n"
        "  --trials  10\n"
        "  --leaf    64  (Strassen leaf; set 1 for pure Strassen)\n"
        "  --out     results.csv\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown arg: " + k);
    }
  }
  return a;
}

static matmul::MatrixKind parse_kind(const std::string& s) {
  if (s == "random") return matmul::MatrixKind::RandomUniform;
  if (s == "symmetric") return matmul::MatrixKind::SymmetricUniform;
  if (s == "hermitian") return matmul::MatrixKind::HermitianUniform;
  throw std::runtime_error("Unknown kind: " + s);
}

static std::vector<std::size_t> sizes_pow2_up_to(std::size_t max_n) {
  if (max_n < 2) return {};
  std::size_t p = 1;
  while ((p << 1) <= max_n) p <<= 1;
  std::vector<std::size_t> v;
  for (std::size_t n = 2; n <= p; n <<= 1) v.push_back(n);
  return v;
}

static double mean(const std::vector<double>& x) {
  return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

static double stddev(const std::vector<double>& x) {
  if (x.size() < 2) return 0.0;
  const double m = mean(x);
  double acc = 0.0;
  for (double v : x) acc += (v - m) * (v - m);
  return std::sqrt(acc / static_cast<double>(x.size() - 1));
}

template <class T>
static double checksum_matrix(const matmul::Matrix<T>& C) {
  double s = 0.0;
  for (std::size_t i = 0; i < C.rows(); ++i) {
    for (std::size_t j = 0; j < C.cols(); ++j) {
      if constexpr (matmul::is_complex_v<T>) s += std::abs(C(i, j));
      else s += static_cast<double>(C(i, j));
    }
  }
  return s;
}

static void ops_naive(std::size_t n, std::uint64_t& mul, std::uint64_t& add) {
  mul = static_cast<std::uint64_t>(n) * n * n;
  add = static_cast<std::uint64_t>(n) * n * (n > 0 ? (n - 1) : 0);
}

static void ops_strassen(std::size_t n, std::size_t leaf, std::uint64_t& mul, std::uint64_t& add) {
  if (n <= leaf) { ops_naive(n, mul, add); return; }
  const std::size_t m = n / 2;
  std::uint64_t m_mul=0, m_add=0;
  ops_strassen(m, leaf, m_mul, m_add);
  mul = 7 * m_mul;
  add = 7 * m_add + 18ull * static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(m);
}

static void real_equiv_ops(bool is_complex,
                           std::uint64_t field_mul, std::uint64_t field_add,
                           std::uint64_t& real_mul, std::uint64_t& real_add) {
  if (!is_complex) { real_mul = field_mul; real_add = field_add; return; }
  real_mul = 4ull * field_mul;
  real_add = 2ull * field_mul + 2ull * field_add;
}

template <class T>
static void bench_one_dtype(const Args& args) {
  const auto kind = parse_kind(args.kind);
  const auto sizes = sizes_pow2_up_to(args.max_n);

  std::ofstream out(args.out, std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("Failed to open output: " + args.out);

  out << "algo,dtype,kind,n,warmups,trials,leaf,mean_s,min_s,std_s,"
         "bytes_abcs,extra_bytes_est,ws_before,ws_after,ws_delta,"
         "field_mul,field_add,field_ops,real_mul_equiv,real_add_equiv,real_ops_equiv,checksum\n";

  for (std::size_t n : sizes) {
    matmul::Matrix<T> A(n, n), B(n, n), C(n, n);

    // Fill once
    matmul::fill_matrix<T>(A.view(), kind, 1000 + static_cast<std::uint64_t>(n));
    matmul::fill_matrix<T>(B.view(), kind, 2000 + static_cast<std::uint64_t>(n));

    // ---- OPTION A FIX STARTS HERE ----
    // Make const refs so Aconst.view() / Bconst.view() produce MatrixView<const T>.
    const matmul::Matrix<T>& Aconst = A;
    const matmul::Matrix<T>& Bconst = B;
    // ---- OPTION A FIX ENDS HERE ----

    const auto mem_before = matmul::get_process_memory_info();

    matmul::StrassenConfig scfg;
    scfg.leaf_size = args.leaf;
    scfg.pad_to_power_of_two = false;

    std::size_t extra_bytes_est = 0;
    std::unique_ptr<matmul::ScratchArena<T>> arena;
    if (args.algo == "strassen") {
      extra_bytes_est = matmul::strassen_scratch_bytes<T>(n);
      arena = std::make_unique<matmul::ScratchArena<T>>(extra_bytes_est);
    }

    // Warmups
    for (int w = 0; w < args.warmups; ++w) {
      if (args.algo == "naive") {
        matmul::gemm_naive<T>(A.view(), B.view(), C.view(), nullptr);
      } else if (args.algo == "strassen") {
        matmul::gemm_strassen_pow2_prealloc<T>(Aconst.view(), Bconst.view(), C.view(), *arena, nullptr, scfg);
      } else {
        throw std::runtime_error("Unknown algo: " + args.algo);
      }
      g_sink = checksum_matrix(C);
    }

    // Timed trials
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(args.trials));
    double final_checksum = 0.0;

    for (int t = 0; t < args.trials; ++t) {
      matmul::Stopwatch sw;
      sw.start();

      if (args.algo == "naive") {
        matmul::gemm_naive<T>(A.view(), B.view(), C.view(), nullptr);
      } else {
        matmul::gemm_strassen_pow2_prealloc<T>(Aconst.view(), Bconst.view(), C.view(), *arena, nullptr, scfg);
      }

      const double sec = sw.stop_seconds();
      times.push_back(sec);

      final_checksum = checksum_matrix(C);
      g_sink = final_checksum;
    }

    const auto mem_after = matmul::get_process_memory_info();

    std::uint64_t field_mul=0, field_add=0;
    if (args.algo == "naive") ops_naive(n, field_mul, field_add);
    else ops_strassen(n, args.leaf, field_mul, field_add);

    const std::uint64_t field_ops = field_mul + field_add;

    std::uint64_t real_mul=0, real_add=0;
    real_equiv_ops(matmul::is_complex_v<T>, field_mul, field_add, real_mul, real_add);
    const std::uint64_t real_ops = real_mul + real_add;

    const double mean_s = mean(times);
    const double min_s = *std::min_element(times.begin(), times.end());
    const double std_s = stddev(times);

    const std::size_t bytes_abcs = 3ull * n * n * sizeof(T);
    const std::size_t ws_before = mem_before.working_set_bytes;
    const std::size_t ws_after  = mem_after.working_set_bytes;
    const std::size_t ws_delta  = (ws_after >= ws_before) ? (ws_after - ws_before) : 0;

    out << args.algo << ","
        << (matmul::is_complex_v<T> ? "complex" : "real") << ","
        << args.kind << ","
        << n << ","
        << args.warmups << ","
        << args.trials << ","
        << args.leaf << ","
        << mean_s << ","
        << min_s << ","
        << std_s << ","
        << bytes_abcs << ","
        << extra_bytes_est << ","
        << ws_before << ","
        << ws_after << ","
        << ws_delta << ","
        << field_mul << ","
        << field_add << ","
        << field_ops << ","
        << real_mul << ","
        << real_add << ","
        << real_ops << ","
        << final_checksum
        << "\n";

    std::cout << "n=" << n << " done. mean_s=" << mean_s << " min_s=" << min_s << "\n";
  }
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    if (args.dtype == "real") {
      bench_one_dtype<double>(args);
    } else if (args.dtype == "complex") {
      bench_one_dtype<std::complex<double>>(args);
    } else {
      throw std::runtime_error("Unknown dtype: " + args.dtype);
    }

    std::cout << "Wrote CSV: " << args.out << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
