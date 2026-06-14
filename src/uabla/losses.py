"""Training-only losses and diagnostics for UABLA routing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .routing import CandidateSelection


def teacher_attention_to_region_targets(
    teacher_attention: torch.Tensor,
    *,
    region_size_tokens: int,
    num_regions: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert dense teacher attention into region-level routing targets.

    teacher_attention may have shape [batch, seq, seq] or [batch, heads, seq, seq].
    This function is training-only; it is not used at inference.
    """

    if teacher_attention.ndim == 4:
        teacher_attention = teacher_attention.mean(dim=1)
    if teacher_attention.ndim != 3:
        raise ValueError("teacher_attention must be [batch, seq, seq] or [batch, heads, seq, seq]")
    if region_size_tokens <= 0:
        raise ValueError("region_size_tokens must be positive")
    teacher_attention = teacher_attention.float()

    batch, query_len, key_len = teacher_attention.shape
    if num_regions is None:
        num_regions = (key_len + region_size_tokens - 1) // region_size_tokens
    padded_key_len = num_regions * region_size_tokens
    pad_len = padded_key_len - key_len
    if pad_len > 0:
        teacher_attention = F.pad(teacher_attention, (0, pad_len))

    targets = teacher_attention.view(batch, query_len, num_regions, region_size_tokens).sum(dim=-1)
    normalizer = targets.sum(dim=-1, keepdim=True).clamp_min(eps)
    return targets / normalizer


def routing_distillation_loss(
    route_scores: torch.Tensor,
    routeable_mask: torch.Tensor,
    teacher_attention: torch.Tensor,
    *,
    region_size_tokens: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Distill dense teacher attention mass into UABLA route probabilities.

    route_scores and routeable_mask are shaped [batch, seq, regions, 4].
    The loss trains block or superblock routing without changing inference.
    """

    if route_scores.shape != routeable_mask.shape:
        raise ValueError("route_scores and routeable_mask must have the same shape")
    route_scores = route_scores.float()
    teacher_attention = teacher_attention.float()
    batch, seq_len, num_regions, centroids_per_region = route_scores.shape
    targets = teacher_attention_to_region_targets(
        teacher_attention,
        region_size_tokens=region_size_tokens,
        num_regions=num_regions,
        eps=eps,
    )
    if targets.shape[:2] != (batch, seq_len):
        raise ValueError("teacher_attention batch/sequence dimensions must match route_scores")

    routeable_regions = routeable_mask.any(dim=-1)
    targets = targets * routeable_regions.to(targets.dtype)
    target_mass = targets.sum(dim=-1, keepdim=True)
    valid = (target_mass.squeeze(-1) > eps) & routeable_regions.any(dim=-1)
    if not bool(valid.any()):
        return route_scores.new_zeros(())
    targets = targets / target_mass.clamp_min(eps)

    flat_scores = route_scores.reshape(batch, seq_len, num_regions * centroids_per_region)
    flat_mask = routeable_mask.reshape(batch, seq_len, num_regions * centroids_per_region)
    flat_scores = flat_scores[valid]
    flat_mask = flat_mask[valid]
    targets = targets[valid]
    safe_scores = flat_scores.masked_fill(~flat_mask, torch.finfo(flat_scores.dtype).min)
    centroid_probs = torch.softmax(safe_scores, dim=-1) * flat_mask.to(flat_scores.dtype)
    centroid_probs = centroid_probs / centroid_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    region_probs = centroid_probs.reshape(
        -1,
        num_regions,
        centroids_per_region,
    ).sum(dim=-1)

    per_token = -(targets * torch.log(region_probs.clamp_min(eps))).sum(dim=-1)
    return per_token.mean()


def direct_routing_supervision_loss(
    route_scores: torch.Tensor,
    routeable_mask: torch.Tensor,
    answer_indices: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    region_size_tokens: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Train query routing toward the region containing each source value.

    This is a synthetic-task-only auxiliary loss. It supervises routing at each
    answer query position, without changing inference behavior.
    """

    if route_scores.shape != routeable_mask.shape:
        raise ValueError("route_scores and routeable_mask must have the same shape")
    if region_size_tokens <= 0:
        raise ValueError("region_size_tokens must be positive")

    route_scores = route_scores.float()
    batch, seq_len, num_regions, centroids_per_region = route_scores.shape
    answer_indices = _ensure_query_dim(answer_indices).to(route_scores.device)
    source_indices = _ensure_query_dim(source_indices).to(route_scores.device)
    if answer_indices.shape != source_indices.shape:
        raise ValueError("answer_indices and source_indices must have the same shape")
    if answer_indices.shape[0] != batch:
        raise ValueError("answer_indices batch dimension must match route_scores")

    query_count = answer_indices.shape[1]
    batch_indices = torch.arange(batch, device=route_scores.device).view(batch, 1)
    batch_indices = batch_indices.expand(batch, query_count)
    answer_indices = answer_indices.clamp(min=0, max=max(seq_len - 1, 0))

    query_scores = route_scores[batch_indices, answer_indices]
    query_mask = routeable_mask[batch_indices, answer_indices]
    source_regions = source_indices // region_size_tokens
    routeable_regions = query_mask.any(dim=-1)

    in_bounds = (source_regions >= 0) & (source_regions < num_regions)
    safe_regions = source_regions.clamp(min=0, max=max(num_regions - 1, 0))
    source_region_routeable = torch.gather(
        routeable_regions,
        dim=-1,
        index=safe_regions.unsqueeze(-1),
    ).squeeze(-1)
    valid = in_bounds & source_region_routeable
    if not bool(valid.any()):
        return route_scores.new_zeros(())

    flat_scores = query_scores.reshape(batch, query_count, num_regions * centroids_per_region)
    flat_mask = query_mask.reshape(batch, query_count, num_regions * centroids_per_region)
    safe_scores = flat_scores.masked_fill(~flat_mask, torch.finfo(flat_scores.dtype).min)
    centroid_probs = torch.softmax(safe_scores, dim=-1) * flat_mask.to(flat_scores.dtype)
    centroid_probs = centroid_probs / centroid_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    region_probs = centroid_probs.reshape(
        batch,
        query_count,
        num_regions,
        centroids_per_region,
    ).sum(dim=-1)

    target_probs = torch.gather(region_probs, dim=-1, index=safe_regions.unsqueeze(-1)).squeeze(-1)
    return -torch.log(target_probs[valid].clamp_min(eps)).mean()


def token_contrastive_retrieval_loss(
    token_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    answer_indices: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Contrastively rank the true source token above other retrieved candidates."""

    if token_scores.shape != candidate_indices.shape or token_scores.shape != candidate_mask.shape:
        raise ValueError("token_scores, candidate_indices, and candidate_mask must have same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    token_scores = token_scores.float()
    batch, seq_len, _ = token_scores.shape
    answer_indices = _ensure_query_dim(answer_indices).to(token_scores.device)
    source_indices = _ensure_query_dim(source_indices).to(token_scores.device)
    if answer_indices.shape != source_indices.shape:
        raise ValueError("answer_indices and source_indices must have the same shape")
    if answer_indices.shape[0] != batch:
        raise ValueError("answer_indices batch dimension must match token_scores")

    query_count = answer_indices.shape[1]
    batch_indices = torch.arange(batch, device=token_scores.device).view(batch, 1)
    batch_indices = batch_indices.expand(batch, query_count)
    answer_indices = answer_indices.clamp(min=0, max=max(seq_len - 1, 0))

    query_scores = token_scores[batch_indices, answer_indices] / temperature
    query_indices = candidate_indices[batch_indices, answer_indices]
    query_mask = candidate_mask[batch_indices, answer_indices]
    positive_mask = query_mask & (query_indices == source_indices.unsqueeze(-1))
    valid = positive_mask.any(dim=-1)
    if not bool(valid.any()):
        return token_scores.new_zeros(())

    safe_scores = query_scores.masked_fill(~query_mask, torch.finfo(query_scores.dtype).min)
    positive_scores = query_scores.masked_fill(
        ~positive_mask,
        torch.finfo(query_scores.dtype).min,
    )
    log_denominator = torch.logsumexp(safe_scores, dim=-1)
    log_positive = torch.logsumexp(positive_scores, dim=-1)
    return (log_denominator[valid] - log_positive[valid]).mean()


def guarded_budget_loss(
    token_budgets: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    prediction_entropy: torch.Tensor | None = None,
    max_budget: int | None = None,
    confidence_threshold: float = 0.55,
    guard_temperature: float = 0.1,
    detach_guard: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize budget mostly when the model is already prediction-confident."""

    if max_budget is None:
        max_budget = int(token_budgets.max().item())
    if max_budget <= 0:
        raise ValueError("max_budget must be positive")
    if prediction_entropy is None:
        if logits is None:
            raise ValueError("either logits or prediction_entropy must be provided")
        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)
        prediction_entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(logits.shape[-1], device=logits.device, dtype=logits.dtype))
        confidence = 1.0 - prediction_entropy / max_entropy.clamp_min(eps)
    else:
        confidence = 1.0 - prediction_entropy

    guard = torch.sigmoid((confidence - confidence_threshold) / guard_temperature)
    if detach_guard:
        guard = guard.detach()
    budget_fraction = token_budgets.to(guard.dtype) / float(max_budget)
    return (guard * budget_fraction).mean()


def routing_diagnostics(
    candidates: CandidateSelection,
    route_scores: torch.Tensor,
    routeable_mask: torch.Tensor,
    *,
    attention: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return compact training diagnostics for router health."""

    batch, seq_len, num_regions, centroids_per_region = route_scores.shape
    route_scores = route_scores.float()
    flat_scores = route_scores.reshape(batch, seq_len, num_regions * centroids_per_region)
    flat_mask = routeable_mask.reshape(batch, seq_len, num_regions * centroids_per_region)
    safe_scores = flat_scores.masked_fill(~flat_mask, torch.finfo(flat_scores.dtype).min)
    probs = torch.softmax(safe_scores, dim=-1) * flat_mask.to(flat_scores.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    valid = flat_mask.any(dim=-1)
    mean_entropy = entropy[valid].mean() if bool(valid.any()) else entropy.sum() * 0.0

    diagnostics = {
        "route_entropy": mean_entropy,
        "avg_open_blocks": candidates.opened_block_mask.sum(dim=-1).float().mean(),
        "avg_centroid_budget": candidates.centroid_budgets.float().mean(),
        "avg_token_budget": candidates.token_budgets.float().mean(),
    }
    if candidates.opened_shifted_block_mask is not None:
        avg_open_shifted = candidates.opened_shifted_block_mask.sum(dim=-1).float().mean()
        diagnostics["avg_open_shifted_blocks"] = avg_open_shifted
        diagnostics["avg_open_total_blocks"] = diagnostics["avg_open_blocks"] + avg_open_shifted
    if candidates.opened_superblock_mask is not None:
        diagnostics["avg_open_superblocks"] = (
            candidates.opened_superblock_mask.sum(dim=-1).float().mean()
        )
    if candidates.opened_shifted_superblock_mask is not None:
        diagnostics["avg_open_shifted_superblocks"] = (
            candidates.opened_shifted_superblock_mask.sum(dim=-1).float().mean()
        )
    if candidates.superblock_budgets is not None:
        diagnostics["avg_superblock_budget"] = candidates.superblock_budgets.float().mean()
    if attention is not None:
        attention = attention.float()
        diagnostics["attention_entropy"] = -(
            attention * torch.log(attention.clamp_min(eps))
        ).sum(dim=-1).mean()
    return diagnostics


def _ensure_query_dim(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(-1)
    if values.ndim == 2:
        return values
    raise ValueError("query index tensors must have shape [batch] or [batch, queries]")
