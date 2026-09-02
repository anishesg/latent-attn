"""Stateless functional API for fused and naive MLA kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from latent_attn import MLAConfig


def _load_ext():
    try:
        from latent_attn import _C
        return _C
    except ImportError as e:
        raise ImportError(
            "latent_attn C extension not found. "
            "Run `pip install -e .` from the repo root to build it."
        ) from e


def fused_mla(
    kv_cache: torch.Tensor,
    q_absorbed: torch.Tensor,
    q_rope: torch.Tensor,
    W_UV: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cfg: "MLAConfig",
) -> torch.Tensor:
    """Single-token fused absorbed MLA forward.

    Args:
        kv_cache:   (seq_len, kv_lora_rank + qk_rope_head_dim) fp16
        q_absorbed: (num_heads, kv_lora_rank) fp32
        q_rope:     (num_heads, qk_rope_head_dim) fp32
        W_UV:       (num_heads * v_head_dim, kv_lora_rank) fp16
        rope_cos:   (seq_len, qk_rope_head_dim // 2) fp32
        rope_sin:   (seq_len, qk_rope_head_dim // 2) fp32
        cfg:        MLAConfig

    Returns:
        (num_heads, v_head_dim) fp32
    """
    _C = _load_ext()
    return _C.fused_mla_forward(
        kv_cache, q_absorbed, q_rope, W_UV, rope_cos, rope_sin,
        cfg.d_model, cfg.kv_lora_rank, cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim, cfg.num_heads, cfg.v_head_dim,
    )


def naive_mla(
    kv_cache: torch.Tensor,
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cfg: "MLAConfig",
) -> torch.Tensor:
    """Single-token naive MLA forward (decompress K/V then attend).

    Args:
        kv_cache: (seq_len, kv_lora_rank + qk_rope_head_dim) fp16
        W_UK:     (num_heads * qk_nope_head_dim, kv_lora_rank) fp16
        W_UV:     (num_heads * v_head_dim, kv_lora_rank) fp16
        q_nope:   (num_heads, qk_nope_head_dim) fp32
        q_rope:   (num_heads, qk_rope_head_dim) fp32
        rope_cos: (seq_len, qk_rope_head_dim // 2) fp32
        rope_sin: (seq_len, qk_rope_head_dim // 2) fp32
        cfg:      MLAConfig

    Returns:
        (num_heads, v_head_dim) fp32
    """
    _C = _load_ext()
    return _C.naive_mla_forward(
        kv_cache, W_UK, W_UV, q_nope, q_rope, rope_cos, rope_sin,
        cfg.d_model, cfg.kv_lora_rank, cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim, cfg.num_heads, cfg.v_head_dim,
    )


def batched_fused_mla(
    kv_cache: torch.Tensor,
    q_absorbed: torch.Tensor,
    q_rope: torch.Tensor,
    W_UV: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cfg: "MLAConfig",
) -> torch.Tensor:
    """Batched fused absorbed MLA forward over multiple tokens sharing one KV cache.

    Args:
        kv_cache:   (seq_len, kv_lora_rank + qk_rope_head_dim) fp16
        q_absorbed: (batch, num_heads, kv_lora_rank) fp32
        q_rope:     (batch, num_heads, qk_rope_head_dim) fp32
        W_UV:       (num_heads * v_head_dim, kv_lora_rank) fp16
        rope_cos:   (seq_len, qk_rope_head_dim // 2) fp32
        rope_sin:   (seq_len, qk_rope_head_dim // 2) fp32
        cfg:        MLAConfig

    Returns:
        (batch, num_heads, v_head_dim) fp32
    """
    _C = _load_ext()
    return _C.batched_fused_mla_forward(
        kv_cache, q_absorbed, q_rope, W_UV, rope_cos, rope_sin,
        cfg.d_model, cfg.kv_lora_rank, cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim, cfg.num_heads, cfg.v_head_dim,
    )
