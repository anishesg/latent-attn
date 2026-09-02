"""Batched correctness test and throughput benchmark for batched_fused_mla.

Compares batched_fused_mla(B tokens) against a loop of single-token fused_mla
calls for B in {1, 4, 16, 64}. Asserts per-token cosine similarity > 0.998.
Throughput benchmark sweeps batch sizes 1..64 reporting tokens/sec.
"""

import math
import sys

import torch
import pytest

from latent_attn import MLAConfig
from latent_attn.functional import fused_mla, batched_fused_mla


def build_rope_tables(seq_len: int, d_R: int, device):
    half = d_R // 2
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    dims = torch.arange(half, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000.0 ** (2.0 * dims / d_R))
    angles = positions[:, None] * freqs[None, :]
    return angles.cos(), angles.sin()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


@pytest.mark.parametrize("batch", [1, 4, 16, 64])
def test_batched_vs_single_loop(batch: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    cfg = MLAConfig.deepseek_v2()
    torch.manual_seed(batch)

    dc = cfg.kv_lora_rank
    dR = cfg.qk_rope_head_dim
    dn = cfg.qk_nope_head_dim
    dv = cfg.v_head_dim
    nh = cfg.num_heads
    S  = 1024

    kv_cache  = torch.randn(S, dc + dR, device=device).to(torch.float16)
    W_UV_fp16 = (torch.randn(nh * dv, dc, device=device) * 0.02).to(torch.float16)
    rope_cos, rope_sin = build_rope_tables(S, dR, device)

    # (B, nh, dc) and (B, nh, dR) absorbed queries
    q_absorbed_b = torch.randn(batch, nh, dc, device=device) * 0.1
    q_rope_b     = torch.randn(batch, nh, dR, device=device) * 0.1

    # Batched kernel
    out_batched = batched_fused_mla(
        kv_cache=kv_cache,
        q_absorbed=q_absorbed_b.contiguous(),
        q_rope=q_rope_b.contiguous(),
        W_UV=W_UV_fp16,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        cfg=cfg,
    )  # (B, nh, dv)

    # Loop of single-token calls
    out_loop = torch.stack([
        fused_mla(
            kv_cache=kv_cache,
            q_absorbed=q_absorbed_b[i].contiguous(),
            q_rope=q_rope_b[i].contiguous(),
            W_UV=W_UV_fp16,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            cfg=cfg,
        )
        for i in range(batch)
    ])  # (B, nh, dv)

    for i in range(batch):
        sim = cosine_sim(out_batched[i], out_loop[i])
        assert sim > 0.998, (
            f"batch={batch}, token={i}: cosine similarity {sim:.6f} < 0.998"
        )

    print(f"\nbatch={batch}: all {batch} tokens passed (min cosine > 0.998)")


def throughput_benchmark(batch_sizes=None, kv_len: int = 2048,
                          warmup: int = 5, iters: int = 30):
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    device = torch.device("cuda")
    cfg = MLAConfig.deepseek_v2()
    torch.manual_seed(0)

    dc = cfg.kv_lora_rank
    dR = cfg.qk_rope_head_dim
    dv = cfg.v_head_dim
    nh = cfg.num_heads
    S  = kv_len

    kv_cache  = torch.randn(S, dc + dR, device=device).to(torch.float16)
    W_UV_fp16 = (torch.randn(nh * dv, dc, device=device) * 0.02).to(torch.float16)
    rope_cos, rope_sin = build_rope_tables(S, dR, device)

    prop = torch.cuda.get_device_properties(0)
    print(f"Device: {prop.name}  kv_len={kv_len}")
    print(f"{'batch':>8}  {'batched_ms':>12}  {'loop_ms':>10}  {'batched_tok/s':>14}  {'loop_tok/s':>12}")
    print("-" * 62)

    for batch in batch_sizes:
        q_abs = torch.randn(batch, nh, dc, device=device) * 0.1
        q_r   = torch.randn(batch, nh, dR, device=device) * 0.1

        # warmup
        for _ in range(warmup):
            batched_fused_mla(kv_cache, q_abs.contiguous(), q_r.contiguous(),
                              W_UV_fp16, rope_cos, rope_sin, cfg)
        torch.cuda.synchronize()

        # time batched
        t_start = torch.cuda.Event(enable_timing=True)
        t_stop  = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(iters):
            batched_fused_mla(kv_cache, q_abs.contiguous(), q_r.contiguous(),
                              W_UV_fp16, rope_cos, rope_sin, cfg)
        t_stop.record()
        torch.cuda.synchronize()
        batched_ms = t_start.elapsed_time(t_stop) / iters

        # time loop of single-token calls
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

        print(f"{batch:>8}  {batched_ms:>12.3f}  {loop_ms:>10.3f}  "
              f"{batched_tok_s:>14.0f}  {loop_tok_s:>12.0f}")


if __name__ == "__main__":
    for batch in [1, 4, 16, 64]:
        test_batched_vs_single_loop(batch)
    print("\nAll batched correctness tests passed.\n")
    throughput_benchmark()
