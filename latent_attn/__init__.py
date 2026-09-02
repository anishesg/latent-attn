"""Fused multi-head latent attention -- absorbed Q/K projection with online softmax."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MLAConfig:
    d_model: int
    kv_lora_rank: int        # d_c: compressed KV latent dimension
    qk_nope_head_dim: int    # per-head non-RoPE dim
    qk_rope_head_dim: int    # per-head RoPE dim (d_R)
    num_heads: int
    v_head_dim: int

    @staticmethod
    def deepseek_v2() -> "MLAConfig":
        return MLAConfig(5120, 512, 128, 64, 128, 128)

    @staticmethod
    def deepseek_v3() -> "MLAConfig":
        return MLAConfig(7168, 512, 128, 64, 128, 128)


def _build_rope_tables(seq_len: int, d_R: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) each of shape (seq_len, d_R // 2), fp32."""
    half = d_R // 2
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # (S,)
    dims = torch.arange(half, dtype=torch.float32, device=device)          # (half,)
    freqs = 1.0 / (10000.0 ** (2.0 * dims / d_R))                          # (half,)
    angles = positions[:, None] * freqs[None, :]                            # (S, half)
    return angles.cos(), angles.sin()


class LatentAttentionLayer(nn.Module):
    """Multi-head latent attention with absorbed W_Q @ W_UK projection.

    The absorbed weight W_QK = W_Q @ W_UK (num_heads, kv_lora_rank, d_model) eliminates
    the decompression step: q_absorbed = x @ W_QK^T is computed once per token, then
    the fused kernel performs attention entirely in the compressed latent space.
    """

    def __init__(self, cfg: MLAConfig, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len

        # Absorbed Q projection: (num_heads, kv_lora_rank, d_model) -> stored as
        # (num_heads * kv_lora_rank, d_model) weight matrix for linear layer
        self.W_Q_absorbed = nn.Parameter(
            torch.empty(cfg.num_heads * cfg.kv_lora_rank, cfg.d_model)
        )
        # RoPE query projection: (num_heads, qk_rope_head_dim, d_model)
        self.W_Q_rope = nn.Parameter(
            torch.empty(cfg.num_heads * cfg.qk_rope_head_dim, cfg.d_model)
        )
        # Key projection (not absorbed, needed for naive path): W_UK
        self.W_UK = nn.Parameter(
            torch.empty(cfg.num_heads * cfg.qk_nope_head_dim, cfg.kv_lora_rank)
        )
        # Value projection W_UV: (num_heads * v_head_dim, kv_lora_rank)
        self.W_UV = nn.Parameter(
            torch.empty(cfg.num_heads * cfg.v_head_dim, cfg.kv_lora_rank)
        )
        # Output projection: (d_model, num_heads * v_head_dim)
        self.W_O = nn.Parameter(
            torch.empty(cfg.d_model, cfg.num_heads * cfg.v_head_dim)
        )

        self._init_weights()

        # RoPE tables -- registered as buffers so they move with .to(device)
        cos, sin = _build_rope_tables(max_seq_len, cfg.qk_rope_head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos)  # (max_seq_len, d_R/2)
        self.register_buffer("rope_sin", sin)  # (max_seq_len, d_R/2)

    def _init_weights(self) -> None:
        cfg = self.cfg
        std = 0.02
        nn.init.normal_(self.W_Q_absorbed, std=std)
        nn.init.normal_(self.W_Q_rope,     std=std)
        nn.init.normal_(self.W_UK,         std=std)
        nn.init.normal_(self.W_UV,         std=std)
        nn.init.normal_(self.W_O,          std=std / math.sqrt(2.0))

    @staticmethod
    def from_config(cfg: MLAConfig, max_seq_len: int = 8192) -> "LatentAttentionLayer":
        return LatentAttentionLayer(cfg, max_seq_len)

    # ------------------------------------------------------------------
    # Helpers to produce fp16 parameter views for CUDA kernels
    # ------------------------------------------------------------------

    def _w_uv_fp16(self) -> torch.Tensor:
        return self.W_UV.detach().to(torch.float16).contiguous()

    def _w_uk_fp16(self) -> torch.Tensor:
        return self.W_UK.detach().to(torch.float16).contiguous()

    # ------------------------------------------------------------------
    # Forward: fused path (single decode token)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,          # (d_model,) or (1, d_model) -- single query token
        kv_cache: torch.Tensor,   # (seq_len, kv_lora_rank + qk_rope_head_dim) fp16
    ) -> torch.Tensor:
        """Fused decode forward. Returns (d_model,) fp32."""
        from latent_attn.functional import fused_mla

        if x.dim() == 2:
            x = x.squeeze(0)
        assert x.dim() == 1 and x.size(0) == self.cfg.d_model

        cfg = self.cfg
        seq_len = kv_cache.size(0)

        # q_absorbed: (num_heads, kv_lora_rank)
        q_abs = (x @ self.W_Q_absorbed.T).view(cfg.num_heads, cfg.kv_lora_rank)
        # q_rope: (num_heads, qk_rope_head_dim)
        q_r   = (x @ self.W_Q_rope.T).view(cfg.num_heads, cfg.qk_rope_head_dim)

        cos = self.rope_cos[:seq_len].contiguous()
        sin = self.rope_sin[:seq_len].contiguous()

        out = fused_mla(
            kv_cache=kv_cache,
            q_absorbed=q_abs.float().contiguous(),
            q_rope=q_r.float().contiguous(),
            W_UV=self._w_uv_fp16(),
            rope_cos=cos,
            rope_sin=sin,
            cfg=cfg,
        )  # (num_heads, v_head_dim)

        # Project back to d_model
        flat = out.reshape(-1)  # (num_heads * v_head_dim,)
        return flat @ self.W_O.T  # (d_model,)

    # ------------------------------------------------------------------
    # Reference forward: naive decompress-then-attend path
    # ------------------------------------------------------------------

    def reference_forward(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-torch reference -- no custom CUDA kernel."""
        if x.dim() == 2:
            x = x.squeeze(0)
        cfg = self.cfg
        seq_len = kv_cache.size(0)

        c_t       = kv_cache[:, :cfg.kv_lora_rank].float()       # (S, dc)
        k_rope_pk = kv_cache[:, cfg.kv_lora_rank:].float()       # (S, dR)

        # Decompress K_nope and V
        W_UK_f = self.W_UK.float()  # (nh*dn, dc)
        W_UV_f = self.W_UV.float()  # (nh*dv, dc)

        K_nope = (c_t @ W_UK_f.T).view(seq_len, cfg.num_heads, cfg.qk_nope_head_dim)
        V      = (c_t @ W_UV_f.T).view(seq_len, cfg.num_heads, cfg.v_head_dim)

        # Apply RoPE to k_rope
        cos = self.rope_cos[:seq_len].float()  # (S, dR/2)
        sin = self.rope_sin[:seq_len].float()
        half = cfg.qk_rope_head_dim // 2
        k0, k1 = k_rope_pk[:, :half], k_rope_pk[:, half:]
        K_rope = torch.cat([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1)  # (S, dR)

        # Query projections
        q_abs_flat = x @ self.W_Q_absorbed.T  # (nh*dc,)
        q_nope = q_abs_flat.view(cfg.num_heads, cfg.kv_lora_rank)
        # For reference path use W_UK to get q_nope in nope dim
        q_nope_proj = (q_nope.float() @ W_UK_f.float()).view(cfg.num_heads, cfg.qk_nope_head_dim)
        q_r_flat = x @ self.W_Q_rope.T
        q_rope_h = q_r_flat.view(cfg.num_heads, cfg.qk_rope_head_dim).float()

        scale = 1.0 / math.sqrt(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)

        # Scores: (nh, S)
        scores_nope = torch.einsum("hd,shd->hs", q_nope_proj, K_nope) * scale
        scores_rope = torch.einsum("hd,sd->hs", q_rope_h, K_rope) * scale
        scores = scores_nope + scores_rope  # (nh, S)

        attn = torch.softmax(scores, dim=-1)  # (nh, S)
        out = torch.einsum("hs,shv->hv", attn, V)  # (nh, v_head_dim)

        flat = out.reshape(-1)
        return flat @ self.W_O.T
