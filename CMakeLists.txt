cmake_minimum_required(VERSION 3.20)
project(matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul src/main.cpp)
target_include_directories(matmul PRIVATE include)

add_executable(matmul_bench src/bench.cpp)
target_include_directories(matmul_bench PRIVATE include)

if (MSVC)
  foreach(tgt IN ITEMS matmul matmul_bench)
    target_compile_options(${tgt} PRIVATE /W4 /permissive- /EHsc /O2)
    target_compile_definitions(${tgt} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(${tgt} PRIVATE psapi)
  endforeach()
else()
  foreach(tgt IN ITEMS matmul matmul_bench)
    target_compile_options(${tgt} PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endforeach()
endif()
