#pragma once

#include <cstddef>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif

  #include <windows.h>
  #include <psapi.h>
#endif

namespace matmul {

struct ProcessMemoryInfo {
  std::size_t working_set_bytes = 0;
  std::size_t peak_working_set_bytes = 0;
  std::size_t pagefile_bytes = 0;
  std::size_t peak_pagefile_bytes = 0;
};

inline ProcessMemoryInfo get_process_memory_info() {
  ProcessMemoryInfo out{};

#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(),
                           reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                           sizeof(pmc))) {
    out.working_set_bytes = static_cast<std::size_t>(pmc.WorkingSetSize);
    out.peak_working_set_bytes = static_cast<std::size_t>(pmc.PeakWorkingSetSize);
    out.pagefile_bytes = static_cast<std::size_t>(pmc.PrivateUsage);
    out.peak_pagefile_bytes = static_cast<std::size_t>(pmc.PeakPagefileUsage);
  }
#endif

  return out;
}

} // namespace matmul
