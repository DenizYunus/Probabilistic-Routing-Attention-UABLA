"""Probabilistic distance functions for UABLA routing."""

from __future__ import annotations

import torch


def cheap_gaussian_distance(
    mu_q: torch.Tensor,
    log_sigma_q: torch.Tensor,
    mu_s: torch.Tensor,
    log_sigma_s: torch.Tensor,
    *,
    alpha: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the locked V1 cheap diagonal Gaussian distance.

    Inputs must be broadcastable and share the same final dimension.
    """

    sigma_q = log_sigma_q.exp()
    sigma_s = log_sigma_s.exp()
    scaled_diff = (mu_q - mu_s) / (sigma_q + sigma_s + eps)
    mean_term = scaled_diff.square().sum(dim=-1)
    sigma_term = (log_sigma_q - log_sigma_s).abs().sum(dim=-1)
    return mean_term + alpha * sigma_term
