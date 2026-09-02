"""Correctness test: fused_mla vs naive_mla and pure-torch reference.

DeepSeek-V2 config: d_c=512, d_R=64, 128 heads.
Sweeps kv_len in {512, 2048, 8192}. Asserts cosine similarity > 0.998.
"""

import math
import sys

import torch
import pytest

from latent_attn import MLAConfig
from latent_attn.functional import fused_mla, naive_mla


def build_rope_tables(seq_len: int, d_R: int, device) -> tuple:
    half = d_R // 2
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    dims = torch.arange(half, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000.0 ** (2.0 * dims / d_R))
    angles = positions[:, None] * freqs[None, :]
    return angles.cos(), angles.sin()


def torch_mla_reference(
    kv_cache: torch.Tensor,   # (S, dc + dR) fp16
    W_UK: torch.Tensor,       # (nh*dn, dc) fp32
    W_UV: torch.Tensor,       # (nh*dv, dc) fp32
    q_nope: torch.Tensor,     # (nh, dn) fp32
    q_rope: torch.Tensor,     # (nh, dR) fp32
    rope_cos: torch.Tensor,   # (S, dR/2) fp32
    rope_sin: torch.Tensor,   # (S, dR/2) fp32
    cfg: MLAConfig,
) -> torch.Tensor:
    """Pure-torch MLA reference -- no custom CUDA kernels."""
    S  = kv_cache.size(0)
    dc = cfg.kv_lora_rank
    dR = cfg.qk_rope_head_dim
    nh = cfg.num_heads
    dn = cfg.qk_nope_head_dim
    dv = cfg.v_head_dim

    c_t   = kv_cache[:, :dc].float()   # (S, dc)
    k_r   = kv_cache[:, dc:].float()   # (S, dR)

    K_nope = (c_t @ W_UK.T).view(S, nh, dn)  # (S, nh, dn)
    V      = (c_t @ W_UV.T).view(S, nh, dv)  # (S, nh, dv)

    # Apply RoPE to k_r
    half = dR // 2
    k0, k1 = k_r[:, :half], k_r[:, half:]
    K_rope = torch.cat([k0 * rope_cos - k1 * rope_sin,
                        k0 * rope_sin + k1 * rope_cos], dim=-1)  # (S, dR)

    scale = 1.0 / math.sqrt(dn + dR)
    scores_nope = torch.einsum("hd,shd->hs", q_nope, K_nope) * scale  # (nh, S)
    scores_rope = torch.einsum("hd,sd->hs",  q_rope, K_rope) * scale  # (nh, S)
    scores = scores_nope + scores_rope

    attn = torch.softmax(scores, dim=-1)        # (nh, S)
    out  = torch.einsum("hs,shv->hv", attn, V)  # (nh, dv)
    return out


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


@pytest.mark.parametrize("kv_len", [512, 2048, 8192])
def test_fused_vs_naive_and_torch(kv_len: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    cfg = MLAConfig.deepseek_v2()
    torch.manual_seed(42)

    dc = cfg.kv_lora_rank
    dR = cfg.qk_rope_head_dim
    dn = cfg.qk_nope_head_dim
    dv = cfg.v_head_dim
    nh = cfg.num_heads
    S  = kv_len

    # Build random tensors
    kv_cache  = torch.randn(S, dc + dR, device=device).to(torch.float16)
    W_UK_fp16 = (torch.randn(nh * dn, dc, device=device) * 0.02).to(torch.float16)
    W_UV_fp16 = (torch.randn(nh * dv, dc, device=device) * 0.02).to(torch.float16)
    q_nope    = torch.randn(nh, dn, device=device) * 0.1
    q_rope    = torch.randn(nh, dR, device=device) * 0.1

    rope_cos, rope_sin = build_rope_tables(S, dR, device)

    # Absorbed query: q_absorbed[h] = q_nope[h] @ W_UK -- shape (nh, dc)
    W_UK_f = W_UK_fp16.float().view(nh, dn, dc)   # (nh, dn, dc)
    q_absorbed = torch.einsum("hd,hdc->hc", q_nope, W_UK_f)  # (nh, dc)

    # --- fused kernel output ---
    out_fused = fused_mla(
        kv_cache=kv_cache,
        q_absorbed=q_absorbed.contiguous(),
        q_rope=q_rope.contiguous(),
        W_UV=W_UV_fp16,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        cfg=cfg,
    )  # (nh, dv)

    # --- naive kernel output ---
    out_naive = naive_mla(
        kv_cache=kv_cache,
        W_UK=W_UK_fp16,
        W_UV=W_UV_fp16,
        q_nope=q_nope.contiguous(),
        q_rope=q_rope.contiguous(),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        cfg=cfg,
    )  # (nh, dv)

    # --- pure-torch reference ---
    out_ref = torch_mla_reference(
        kv_cache, W_UK_fp16.float(), W_UV_fp16.float(),
        q_nope, q_rope, rope_cos, rope_sin, cfg,
    )

    sim_fn = cosine_sim(out_fused, out_naive)
    sim_fr = cosine_sim(out_fused, out_ref)
    sim_nr = cosine_sim(out_naive, out_ref)

    print(f"\nkv_len={kv_len}: fused/naive={sim_fn:.6f}  fused/ref={sim_fr:.6f}  naive/ref={sim_nr:.6f}")

    assert sim_fn > 0.998, f"fused vs naive cosine={sim_fn:.6f} at kv_len={kv_len}"
    assert sim_fr > 0.998, f"fused vs ref  cosine={sim_fr:.6f} at kv_len={kv_len}"
    assert sim_nr > 0.998, f"naive vs ref  cosine={sim_nr:.6f} at kv_len={kv_len}"


if __name__ == "__main__":
    for kv_len in [512, 2048, 8192]:
        test_fused_vs_naive_and_torch(kv_len)
    print("All correctness tests passed.")
