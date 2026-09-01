#pragma once
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "mla_config.cuh"

// Error checking macros
#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t status = (x); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status); \
        exit(1); \
    } \
} while(0)

// Workspace for naive MLA: decompressed K/V buffers in fp32 for accumulation.
struct NaiveMlaWorkspace {
    float* K_nope;   // (seq_len, num_heads, qk_nope_head_dim)
    float* K_rope;   // (seq_len, qk_rope_head_dim) -- RoPE-rotated copy
    float* V;        // (seq_len, num_heads, v_head_dim)
    float* output;   // (num_heads, v_head_dim)

    int seq_len;
    const MLAConfig& cfg;

    NaiveMlaWorkspace(int sl, const MLAConfig& c);
    ~NaiveMlaWorkspace();
};

// Decompress K and V from latent cache into full-size fp32 buffers.
// W_UK: (num_heads * qk_nope_head_dim, kv_lora_rank) fp16, row-major
// W_UV: (num_heads * v_head_dim,       kv_lora_rank) fp16, row-major
// c_t:  (seq_len, kv_lora_rank) fp16
// k_rope_packed: (seq_len, qk_rope_head_dim) fp16 (from latent cache)
// RoPE cos/sin: (max_seq_len, qk_rope_head_dim/2) fp32
void naive_decompress_kv(
    cublasHandle_t handle,
    const __half* c_t,           // (seq_len, kv_lora_rank)
    const __half* k_rope_packed, // (seq_len, qk_rope_head_dim)
    const __half* W_UK,          // (num_heads * qk_nope_head_dim, kv_lora_rank)
    const __half* W_UV,          // (num_heads * v_head_dim,       kv_lora_rank)
    const float*  rope_cos,      // (max_seq_len, qk_rope_head_dim/2)
    const float*  rope_sin,      // (max_seq_len, qk_rope_head_dim/2)
    NaiveMlaWorkspace& ws,
    const MLAConfig& cfg,
    cudaStream_t stream = 0);

// Run standard multi-head attention over decompressed full-size K/V.
// q_nope: (num_heads, qk_nope_head_dim) fp32, one query token
// q_rope: (num_heads, qk_rope_head_dim) fp32
// Writes result to ws.output: (num_heads, v_head_dim) fp32
void naive_attention(
    const float* q_nope,   // (num_heads, qk_nope_head_dim)
    const float* q_rope,   // (num_heads, qk_rope_head_dim)
    NaiveMlaWorkspace& ws,
    const MLAConfig& cfg,
    cudaStream_t stream = 0);

// Convenience wrapper: decompress + attend, writes to out_fp32 (num_heads, v_head_dim)
void naive_mla(
    cublasHandle_t handle,
    const __half* c_t,
    const __half* k_rope_packed,
    const __half* W_UK,
    const __half* W_UV,
    const float*  q_nope,
    const float*  q_rope,
    const float*  rope_cos,
    const float*  rope_sin,
    float*        out_fp32,
    NaiveMlaWorkspace& ws,
    const MLAConfig& cfg,
    cudaStream_t stream = 0);
