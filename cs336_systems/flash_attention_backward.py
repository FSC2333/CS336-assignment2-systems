from __future__ import annotations

import math

import torch


@torch.compile
def flash_attention_backward_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    if is_causal:
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        causal_mask = torch.arange(n_queries, device=Q.device)[:, None] >= torch.arange(n_keys, device=Q.device)[None, :]
        scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e6))

    P = torch.exp(scores - L[..., None])
    D_vec = torch.sum(dO * output, dim=-1)

    dV = torch.matmul(P.transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    dS = P * (dP - D_vec[..., None])
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-1, -2), Q) * scale

    return dQ, dK, dV
