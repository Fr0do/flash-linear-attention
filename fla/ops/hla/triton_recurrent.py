# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
import triton
import triton.language as tl
from einops import repeat


@triton.jit
def _recurrent_hla_fwd_kernel(
    q,
    k,
    v,
    o,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EPS: tl.constexpr,
    RIDGE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh - b * H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    v_start = pid_v * BLOCK_V
    mask_k = offs_k < K
    mask_v = (v_start + offs_v) < V

    S = tl.zeros((BLOCK_K, BLOCK_K), dtype=tl.float32)
    C = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    m = tl.zeros((BLOCK_K,), dtype=tl.float32)
    G = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    h_state = tl.zeros((BLOCK_K,), dtype=tl.float32)

    q_bh = (b * T * H + h) * K
    v_bh = (b * T * H + h) * V
    for t in range(0, T):
        q_t = tl.load(q + q_bh + t * H * K + offs_k, mask=mask_k, other=0.0).to(tl.float32) * SCALE
        k_t = tl.load(k + q_bh + t * H * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v + v_bh + t * H * V + v_start + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        C_prev = C
        m_prev = m

        dS = k_t[:, None] * k_t[None, :]
        C = C + q_t[:, None] * v_t[None, :]
        m = m + q_t
        G = G + tl.dot(dS, C_prev, input_precision="ieee")
        h_state = h_state + tl.sum(dS * m_prev[None, :], axis=1)
        S = S + dS

        u = tl.sum(q_t[:, None] * S, axis=0) + RIDGE * q_t
        out = tl.sum(u[:, None] * C, axis=0) - tl.sum(q_t[:, None] * G, axis=0)
        if NORMALIZE:
            den = tl.sum(u * m, axis=0) - tl.sum(q_t * h_state, axis=0)
            den_abs = tl.maximum(tl.abs(den), EPS)
            den_safe = tl.where(den < 0.0, -den_abs, den_abs)
            out = out / den_safe

        tl.store(o + v_bh + t * H * V + v_start + offs_v, out, mask=mask_v)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def triton_recurrent_hla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    normalize: bool = False,
    eps: float = 1e-6,
    ridge: float = 0.0,
    scale: float | None = None,
    max_block_v: int = 32,
) -> tuple[torch.Tensor, None]:
    r"""Triton forward kernel for recurrent second-order HLA.

    This kernel keeps the HLA prefix state inside one Triton program per
    ``(batch, head, value-block)``. It is intended for small per-head key
    dimensions; for larger head dimensions use ``recurrent_hla`` or a future
    chunked kernel.
    """
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("triton_recurrent_hla requires CUDA tensors")
    if initial_state is not None or output_final_state:
        raise NotImplementedError("triton_recurrent_hla does not yet support cached recurrent state output")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must all be rank-4 tensors")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError(f"q/k and v must share [B, T], got {q.shape[:2]} and {v.shape[:2]}")
    if v.shape[2] % q.shape[2] != 0:
        raise ValueError(f"value heads ({v.shape[2]}) must be divisible by q/k heads ({q.shape[2]})")

    if v.shape[2] != q.shape[2]:
        groups = v.shape[2] // q.shape[2]
        q = repeat(q, "b t h d -> b t (h g) d", g=groups)
        k = repeat(k, "b t h d -> b t (h g) d", g=groups)

    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if key_dim > 32:
        raise ValueError("triton_recurrent_hla currently supports key_dim <= 32")
    if value_dim <= 0:
        raise ValueError("value_dim must be positive")
    if scale is None:
        scale = key_dim**-0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(v)

    block_k = max(16, _next_power_of_2(key_dim))
    block_v = max(16, min(max_block_v, _next_power_of_2(value_dim)))
    grid = (batch * heads, triton.cdiv(value_dim, block_v))
    _recurrent_hla_fwd_kernel[grid](
        q,
        k,
        v,
        output,
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        SCALE=float(scale),
        NORMALIZE=normalize,
        EPS=float(eps),
        RIDGE=float(ridge),
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=4,
    )
    return output, None
