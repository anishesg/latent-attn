#pragma once
#include <cuda_fp16.h>
#include "mla_config.cuh"

// Tile size for latent cache streaming through shared memory
#ifndef TILE_SEQ
#define TILE_SEQ 32
#endif

// Fused absorbed MLA decode kernel launch parameters
struct FusedMLAParams {
    // Latent cache: (seq_len, kv_lora_rank + qk_rope_head_dim) fp16
    const __half* kv_cache;   // LatentKVCache::data
    int           seq_len;

    // Absorbed query vectors (precomputed as q_nope @ W_UK): (num_heads, kv_lora_rank) fp32
    const float*  q_absorbed;

    // RoPE query: (num_heads, qk_rope_head_dim) fp32
    const float*  q_rope;

    // Output projection W_UV: (num_heads * v_head_dim, kv_lora_rank) fp16
    // Used to project latent value accumulator to output head dimension
    const __half* W_UV;

    // RoPE tables: (seq_len, qk_rope_head_dim/2) fp32
    const float*  rope_cos;
    const float*  rope_sin;

    // Output: (num_heads, v_head_dim) fp32
    float*        output;

    const MLAConfig* cfg;
};

// Device functions (defined in fused_mla.cu, called from kernel)
__device__ float latent_dot(
    const float* __restrict__ q_abs,   // (kv_lora_rank,) in registers
    const __half* __restrict__ c_tile, // tile row in shared memory
    int kv_lora_rank);

__device__ float rope_score_correction(
    const float* __restrict__ q_rope,
    const __half* __restrict__ k_rope_tile,
    const float* __restrict__ cos_row,
    const float* __restrict__ sin_row,
    int qk_rope_head_dim);

// Host launch function -- computes required shared memory and launches kernel
// Returns cudaSuccess or error code
cudaError_t launch_fused_mla(const FusedMLAParams& p, cudaStream_t stream = 0);
