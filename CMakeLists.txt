cmake_minimum_required(VERSION 3.20)
project(latent_attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Static library
add_library(latent_attn STATIC
    src/naive_mla.cu
    src/fused_mla.cu
)

target_include_directories(latent_attn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(latent_attn PUBLIC
    CUDA::cublas
    CUDA::cudart
)

target_compile_options(latent_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

# Test executable
add_executable(test_correctness tests/test_correctness.cu)
target_include_directories(test_correctness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_correctness PRIVATE latent_attn CUDA::cublas CUDA::cudart)
target_compile_options(test_correctness PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

# Benchmark executable
add_executable(bench_latency benchmarks/bench_latency.cu)
target_include_directories(bench_latency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(bench_latency PRIVATE latent_attn CUDA::cublas CUDA::cudart)
target_compile_options(bench_latency PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)
