#pragma once
#include <cuda_fp16.h>
#include "mla_config.cuh"

// Launch parameters for the batched fused MLA kernel.
// Each token in the batch independently attends over the shared KV cache.
struct BatchedFusedMLAParams {
    // Shared latent KV cache: (seq_len, kv_lora_rank + qk_rope_head_dim) fp16
    const __half* kv_cache;
    int           seq_len;
    int           batch_size;

    // Per-token absorbed query: (batch_size, num_heads, kv_lora_rank) fp32
    const float*  q_absorbed;

    // Per-token RoPE query: (batch_size, num_heads, qk_rope_head_dim) fp32
    const float*  q_rope;

    // Shared output projection W_UV: (num_heads * v_head_dim, kv_lora_rank) fp16
    const __half* W_UV;

    // Shared RoPE tables: (seq_len, qk_rope_head_dim/2) fp32
    const float*  rope_cos;
    const float*  rope_sin;

    // Output: (batch_size, num_heads, v_head_dim) fp32
    float*        output;

    const MLAConfig* cfg;
};

// Host launch function. Grid = (num_heads, batch_size), block = (THREADS_PER_BLOCK,).
// Each block handles one (head, token) pair. RoPE cos/sin tables are cached in
// shared memory once per block to amortize global memory access across the seq loop.
cudaError_t launch_batched_fused_mla(const BatchedFusedMLAParams& p, cudaStream_t stream = 0);
