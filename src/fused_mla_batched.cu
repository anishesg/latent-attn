#include "fused_mla_batched.cuh"
#include "fused_mla.cuh"   // for latent_dot, rope_score_correction
#include <cfloat>
#include <cmath>

// Each block handles one (head h, token b) pair.
// Grid:  (num_heads, batch_size)
// Block: (BATCHED_THREADS,)
//
// Shared memory layout:
//   [0 .. TILE_SEQ * entry_stride * sizeof(__half))   -- fp16 KV tile
//   [tile_bytes .. + kv_lora_rank * sizeof(float))    -- latent value accumulator (acc_lat)
//
// RoPE tables are read directly from global memory; at DeepSeek-V2 config
// (seq_len <= 32768, dR/2=32) the cos/sin tables are 32768*32*4 = 4 MB each --
// too large for shared memory. Instead each thread loads only the rows it needs.

#define BATCHED_THREADS 128

#ifndef TILE_SEQ
#define TILE_SEQ 32
#endif

__global__ void batched_fused_mla_kernel(
    const __half* __restrict__ kv_cache,    // (S, entry_stride) fp16
    int           seq_len,
    int           entry_stride,             // kv_lora_rank + qk_rope_head_dim
    const float*  __restrict__ q_absorbed,  // (B, nh, dc) fp32
    const float*  __restrict__ q_rope,      // (B, nh, dR) fp32
    const __half* __restrict__ W_UV,        // (nh * dv, dc) fp16
    const float*  __restrict__ rope_cos,    // (S, dR/2) fp32
    const float*  __restrict__ rope_sin,    // (S, dR/2) fp32
    float*        __restrict__ output,      // (B, nh, dv) fp32
    int kv_lora_rank,
    int qk_rope_head_dim,
    int v_head_dim,
    int num_heads,
    float scale)
{
    int h = blockIdx.x;   // head index
    int b = blockIdx.y;   // batch (token) index
    int tid = threadIdx.x;
    int half_rope = qk_rope_head_dim / 2;

    // Shared memory
    extern __shared__ char smem_raw[];
    __half* kv_tile = reinterpret_cast<__half*>(smem_raw);
    int tile_fp16_words = TILE_SEQ * entry_stride;
    int tile_bytes = (tile_fp16_words * (int)sizeof(__half) + 3) & ~3;
    float* acc_lat = reinterpret_cast<float*>(smem_raw + tile_bytes);

    // Zero accumulator
    for (int i = tid; i < kv_lora_rank; i += BATCHED_THREADS)
        acc_lat[i] = 0.0f;
    __syncthreads();

    // Query pointers for this (b, h)
    const float* qa = q_absorbed + (b * num_heads + h) * kv_lora_rank;
    const float* qr = q_rope     + (b * num_heads + h) * qk_rope_head_dim;

    // Two-pass softmax (thread 0 only -- see fused_mla.cu for design rationale)
    if (tid == 0) {
        float m = -FLT_MAX;

        // Pass 1: find max score
        for (int pos = 0; pos < seq_len; ++pos) {
            const __half* ct     = kv_cache + pos * entry_stride;
            const __half* k_rope = ct + kv_lora_rank;
            const float*  cos_row = rope_cos + pos * half_rope;
            const float*  sin_row = rope_sin + pos * half_rope;

            float score = latent_dot(qa, ct, kv_lora_rank) * scale
                        + rope_score_correction(qr, k_rope, cos_row, sin_row, qk_rope_head_dim) * scale;
            m = fmaxf(m, score);
        }

        // Pass 2: weighted accumulation
        float l = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            const __half* ct     = kv_cache + pos * entry_stride;
            const __half* k_rope = ct + kv_lora_rank;
            const float*  cos_row = rope_cos + pos * half_rope;
            const float*  sin_row = rope_sin + pos * half_rope;

            float score = latent_dot(qa, ct, kv_lora_rank) * scale
                        + rope_score_correction(qr, k_rope, cos_row, sin_row, qk_rope_head_dim) * scale;
            float w = expf(score - m);
            l += w;
            for (int i = 0; i < kv_lora_rank; ++i)
                acc_lat[i] += w * __half2float(ct[i]);
        }

        // Normalize
        for (int i = 0; i < kv_lora_rank; ++i)
            acc_lat[i] /= l;

        // Project via W_UV
        float* out = output + (b * num_heads + h) * v_head_dim;
        for (int j = 0; j < v_head_dim; ++j) {
            const __half* wuv_j = W_UV + (h * v_head_dim + j) * kv_lora_rank;
            float val = 0.0f;
            for (int i = 0; i < kv_lora_rank; ++i)
                val += acc_lat[i] * __half2float(wuv_j[i]);
            out[j] = val;
        }
    }
}

cudaError_t launch_batched_fused_mla(const BatchedFusedMLAParams& p, cudaStream_t stream)
{
    const MLAConfig& cfg = *p.cfg;
    int entry_stride = cfg.kv_lora_rank + cfg.qk_rope_head_dim;

    int tile_fp16_bytes = (TILE_SEQ * entry_stride * (int)sizeof(__half) + 3) & ~3;
    int smem_bytes = tile_fp16_bytes + cfg.kv_lora_rank * (int)sizeof(float);

    float scale = 1.0f / sqrtf((float)(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim));

    dim3 grid(cfg.num_heads, p.batch_size);
    batched_fused_mla_kernel<<<grid, BATCHED_THREADS, smem_bytes, stream>>>(
        p.kv_cache, p.seq_len, entry_stride,
        p.q_absorbed, p.q_rope,
        p.W_UV,
        p.rope_cos, p.rope_sin,
        p.output,
        cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.num_heads,
        scale);

    return cudaGetLastError();
}
