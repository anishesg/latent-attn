"""Batched throughput benchmark: tokens/sec vs batch size at DeepSeek-V2 config.

Sweeps batch sizes 1..64 at kv_len in {512, 2048, 8192}, comparing
batched_fused_mla against a loop of single-token fused_mla calls.
"""

import sys

import torch

from latent_attn import MLAConfig
from latent_attn.functional import fused_mla, batched_fused_mla


def build_rope_tables(seq_len: int, d_R: int, device):
    half = d_R // 2
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    dims = torch.arange(half, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000.0 ** (2.0 * dims / d_R))
    angles = positions[:, None] * freqs[None, :]
    return angles.cos(), angles.sin()


def run_sweep(cfg: MLAConfig, kv_len: int, batch_sizes, warmup: int = 5, iters: int = 30):
    device = torch.device("cuda")
    dc = cfg.kv_lora_rank
    dR = cfg.qk_rope_head_dim
    dv = cfg.v_head_dim
    nh = cfg.num_heads

    kv_cache  = torch.randn(kv_len, dc + dR, device=device).to(torch.float16)
    W_UV_fp16 = (torch.randn(nh * dv, dc, device=device) * 0.02).to(torch.float16)
    rope_cos, rope_sin = build_rope_tables(kv_len, dR, device)

    t_start = torch.cuda.Event(enable_timing=True)
    t_stop  = torch.cuda.Event(enable_timing=True)

    print(f"\n  kv_len={kv_len}")
    print(f"  {'batch':>8}  {'batched_ms':>12}  {'loop_ms':>10}  "
          f"{'batched_tok/s':>14}  {'loop_tok/s':>12}  {'speedup':>8}")
    print("  " + "-" * 72)

    for batch in batch_sizes:
        q_abs = torch.randn(batch, nh, dc, device=device) * 0.1
        q_r   = torch.randn(batch, nh, dR, device=device) * 0.1

        for _ in range(warmup):
            batched_fused_mla(kv_cache, q_abs.contiguous(), q_r.contiguous(),
                              W_UV_fp16, rope_cos, rope_sin, cfg)
        torch.cuda.synchronize()

        t_start.record()
        for _ in range(iters):
            batched_fused_mla(kv_cache, q_abs.contiguous(), q_r.contiguous(),
                              W_UV_fp16, rope_cos, rope_sin, cfg)
        t_stop.record()
        torch.cuda.synchronize()
        batched_ms = t_start.elapsed_time(t_stop) / iters

        t_start.record()
        for _ in range(iters):
            for i in range(batch):
                fused_mla(kv_cache, q_abs[i].contiguous(), q_r[i].contiguous(),
                          W_UV_fp16, rope_cos, rope_sin, cfg)
        t_stop.record()
        torch.cuda.synchronize()
        loop_ms = t_start.elapsed_time(t_stop) / iters

        batched_tok_s = batch * 1000.0 / batched_ms
        loop_tok_s    = batch * 1000.0 / loop_ms
        speedup       = batched_tok_s / loop_tok_s

        print(f"  {batch:>8}  {batched_ms:>12.3f}  {loop_ms:>10.3f}  "
              f"{batched_tok_s:>14.0f}  {loop_tok_s:>12.0f}  {speedup:>7.2f}x")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- skipping")
        sys.exit(0)

    cfg = MLAConfig.deepseek_v2()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    kv_lens = [512, 2048, 8192]

    prop = torch.cuda.get_device_properties(0)
    peak_bw = 2.0 * prop.memory_clock_rate * 1e3 * prop.memory_bus_width / 8.0 / 1e9
    print(f"Device: {prop.name}  Peak BW: {peak_bw:.1f} GB/s")
    print(f"Config: d_c={cfg.kv_lora_rank}, d_R={cfg.qk_rope_head_dim}, "
          f"{cfg.num_heads} heads, v_dim={cfg.v_head_dim}")

    for kv_len in kv_lens:
        run_sweep(cfg, kv_len, batch_sizes)

    print()


if __name__ == "__main__":
    main()
