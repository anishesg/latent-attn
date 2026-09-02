"""Python CUDA-event latency benchmark for fused vs naive MLA.

DeepSeek-V2 config. Reports wall-clock latency, bytes loaded, effective
bandwidth, and decode tokens/sec across kv_len in {512, 2048, 8192, 32768}.
"""

import math
import sys

import torch

from latent_attn import MLAConfig
from latent_attn.functional import fused_mla, naive_mla


def build_rope_tables(seq_len: int, d_R: int, device):
    half = d_R // 2
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    dims = torch.arange(half, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000.0 ** (2.0 * dims / d_R))
    angles = positions[:, None] * freqs[None, :]
    return angles.cos(), angles.sin()


class CudaTimer:
    def __init__(self):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.stop_ev  = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_ev.record()
        return self

    def __exit__(self, *_):
        self.stop_ev.record()
        torch.cuda.synchronize()

    def elapsed_ms(self) -> float:
        return self.start_ev.elapsed_time(self.stop_ev)


def benchmark(cfg: MLAConfig, kv_len: int, warmup: int = 5, iters: int = 30):
    device = torch.device("cuda")
    torch.manual_seed(0)

    dc = cfg.kv_lora_rank
    dR = cfg.qk_rope_head_dim
    dn = cfg.qk_nope_head_dim
    dv = cfg.v_head_dim
    nh = cfg.num_heads
    S  = kv_len

    kv_cache  = torch.randn(S, dc + dR, device=device).to(torch.float16)
    W_UK_fp16 = (torch.randn(nh * dn, dc, device=device) * 0.02).to(torch.float16)
    W_UV_fp16 = (torch.randn(nh * dv, dc, device=device) * 0.02).to(torch.float16)
    q_nope    = torch.randn(nh, dn, device=device)
    q_rope    = torch.randn(nh, dR, device=device)

    rope_cos, rope_sin = build_rope_tables(S, dR, device)

    W_UK_f = W_UK_fp16.float().view(nh, dn, dc)
    q_absorbed = torch.einsum("hd,hdc->hc", q_nope, W_UK_f).contiguous()

    # warmup
    for _ in range(warmup):
        fused_mla(kv_cache, q_absorbed, q_rope.contiguous(), W_UV_fp16, rope_cos, rope_sin, cfg)
        naive_mla(kv_cache, W_UK_fp16, W_UV_fp16, q_nope.contiguous(), q_rope.contiguous(), rope_cos, rope_sin, cfg)
    torch.cuda.synchronize()

    # time fused
    fused_total_ms = 0.0
    for _ in range(iters):
        with CudaTimer() as t:
            fused_mla(kv_cache, q_absorbed, q_rope.contiguous(), W_UV_fp16, rope_cos, rope_sin, cfg)
        fused_total_ms += t.elapsed_ms()

    # time naive
    naive_total_ms = 0.0
    for _ in range(iters):
        with CudaTimer() as t:
            naive_mla(kv_cache, W_UK_fp16, W_UV_fp16, q_nope.contiguous(), q_rope.contiguous(), rope_cos, rope_sin, cfg)
        naive_total_ms += t.elapsed_ms()

    fused_ms = fused_total_ms / iters
    naive_ms = naive_total_ms / iters

    # Bytes loaded
    # Fused: 2 passes over kv_cache (find max + accumulate)
    fused_bytes  = 2 * S * (dc + dR) * 2  # fp16
    # Naive: c_t + k_rope read; K_nope + K_rope + V written then read
    naive_bytes  = (S * dc * 2                             # read c_t fp16
                  + S * dR * 2                             # read k_rope fp16
                  + S * nh * dn * 4                        # write K_nope fp32
                  + S * nh * dv * 4                        # write V fp32
                  + S * dR * 4                             # write K_rope fp32
                  + S * nh * dn * 4                        # read K_nope
                  + S * dR * 4                             # read K_rope
                  + S * nh * dv * 4)                       # read V

    fused_bw_gb  = fused_bytes  / (fused_ms  * 1e6)
    naive_bw_gb  = naive_bytes  / (naive_ms  * 1e6)
    speedup      = naive_ms / fused_ms
    fused_tok_s  = 1000.0 / fused_ms
    naive_tok_s  = 1000.0 / naive_ms

    return {
        "kv_len":       kv_len,
        "fused_ms":     fused_ms,
        "naive_ms":     naive_ms,
        "speedup":      speedup,
        "fused_bw_gb":  fused_bw_gb,
        "naive_bw_gb":  naive_bw_gb,
        "fused_bytes":  fused_bytes,
        "naive_bytes":  naive_bytes,
        "fused_tok_s":  fused_tok_s,
        "naive_tok_s":  naive_tok_s,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- skipping benchmark")
        sys.exit(0)

    cfg = MLAConfig.deepseek_v2()
    kv_lens = [512, 2048, 8192, 32768]

    prop = torch.cuda.get_device_properties(0)
    peak_bw = 2.0 * prop.memory_clock_rate * 1e3 * prop.memory_bus_width / 8.0 / 1e9
    print(f"Device: {prop.name}  Peak BW: {peak_bw:.1f} GB/s")
    print(f"Config: d_c={cfg.kv_lora_rank}, d_R={cfg.qk_rope_head_dim}, "
          f"{cfg.num_heads} heads, v_dim={cfg.v_head_dim}")
    print()

    hdr = (f"{'kv_len':>8}  {'naive_ms':>10}  {'fused_ms':>10}  "
           f"{'speedup':>8}  {'naive_bw':>10}  {'fused_bw':>10}  "
           f"{'naive_tok/s':>12}  {'fused_tok/s':>12}")
    print(hdr)
    print("-" * len(hdr))

    for kv_len in kv_lens:
        r = benchmark(cfg, kv_len)
        print(f"{r['kv_len']:>8}  {r['naive_ms']:>10.3f}  {r['fused_ms']:>10.3f}  "
              f"{r['speedup']:>7.2f}x  {r['naive_bw_gb']:>9.1f}G  {r['fused_bw_gb']:>9.1f}G  "
              f"{r['naive_tok_s']:>12.0f}  {r['fused_tok_s']:>12.0f}")

    print()
    print("Bytes loaded per decode step:")
    for kv_len in kv_lens:
        r = benchmark(cfg, kv_len, warmup=2, iters=10)
        print(f"  kv_len={kv_len:6d}  naive={r['naive_bytes']//1024:6d} KB  "
              f"fused={r['fused_bytes']//1024:5d} KB  "
              f"reduction={r['naive_bytes']/r['fused_bytes']:.1f}x")


if __name__ == "__main__":
    main()
