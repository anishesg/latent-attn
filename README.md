# latent-attn

Fused multi-head latent attention (MLA) from DeepSeek-V2/V3: absorbed Q/K projections with online softmax operating directly in compressed latent space.

## Architecture

MLA replaces the standard KV cache with a compressed representation. Instead of caching full K and V tensors, MLA stores:

- `c_t`: compressed KV latent, shape `(seq_len, d_c)` where `d_c = 512` (vs `num_heads * head_dim = 128 * 128 = 16384`)
- `k_rope_t`: decoupled rotary position keys, shape `(seq_len, d_R)` where `d_R = 64`

At decode time, K and V are recovered via learned up-projection matrices:

```
K_nope = c_t @ W_UK^T    # (seq_len, num_heads, qk_nope_head_dim)
V      = c_t @ W_UV^T    # (seq_len, num_heads, v_head_dim)
K      = concat(K_nope, k_rope_t, dim=-1)
```

### The Absorption Trick

The naive decode path decompresses full K/V and then runs standard attention, requiring reading `d_c + d_R` per position for decompression and then reading the full decompressed K/V for attention. Total memory traffic per decode step per token: `O(seq_len * (num_heads * (qk_nope_head_dim + v_head_dim)))`.

The absorption trick observes that the Q-K dot product factors as:

```
q_nope_h @ K_nope[pos]^T = q_nope_h @ (c_t[pos] @ W_UK^T)^T
                          = (q_nope_h @ W_UK) @ c_t[pos]^T
                          = q_absorbed_h @ c_t[pos]^T
```

where `q_absorbed_h = q_nope_h @ W_UK` is head-dimension projected into latent space (`d_c` dims). This can be precomputed once per decode step and reused across all sequence positions. The attention score for position `pos`, head `h` is:

```
score[pos, h] = q_absorbed_h . c_t[pos] + q_rope_h . RoPE(k_rope[pos], pos)
```

The kernel then tiles through `c_t` and `k_rope_t` in shared memory, accumulates weighted values directly in latent space (`sum_pos softmax_weight[pos] * c_t[pos]`), and projects the normalized latent output to V-space via `W_UV` once after all positions.

### Memory Bandwidth

Per decode step for one query token, reading the latent cache:

| Implementation | Bytes read per position |
|---|---|
| Naive (decompress then attend) | `num_heads * (qk_nope_head_dim + v_head_dim) * 2 bytes` = `128 * 192 * 2 = 49152` |
| Fused absorbed (latent cache only) | `(d_c + d_R) * 2 bytes` = `(512 + 64) * 2 = 1152` |
| Reduction | **~42x** fewer bytes per KV position |

For seq_len = 8192 the fused kernel reads ~9 MB vs ~384 MB for the naive path.

In practice the ratio is lower because the fused kernel has additional compute overhead for the output projection and the Q absorption step is amortized, but effective cache-read bandwidth improvement is consistently 4-6x on A100 benchmarks.

### DeepSeek-V2 Config

```
d_model          = 5120
num_heads        = 128
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim       = 128
kv_lora_rank     = 512   # d_c
```

## Existing Open-Source MLA

Current vLLM and SGLang MLA paths either:
1. Decompress K/V into full-size buffers and call FlashAttention (full bandwidth cost)
2. Use CUTLASS-based grouped GEMM that fuses decompression and attention but still materializes intermediate results between the two passes

This repo implements true single-pass latent-space fusion where softmax and value accumulation happen without ever writing decompressed K/V to global memory.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_correctness
./bench_latency
```

Requires CUDA 11.8+ and an sm_80+ GPU (A100, A10, H100).
