from __future__ import annotations

import torch


def recall_at_k(hit_mask: torch.Tensor) -> float:
    if hit_mask.numel() == 0:
        return 0.0
    return float(hit_mask.float().mean().item())


def mrr_at_k(ranks: torch.Tensor) -> float:
    if ranks.numel() == 0:
        return 0.0
    ranks_f = ranks.float()
    rr = torch.where(ranks_f > 0, 1.0 / ranks_f, torch.zeros_like(ranks_f))
    return float(rr.mean().item())
