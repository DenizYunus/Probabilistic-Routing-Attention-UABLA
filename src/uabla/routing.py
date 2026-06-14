"""Routing utilities for UABLA-448 V1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import UABLAConfig
from .distance import cheap_gaussian_distance


@dataclass(frozen=True)
class BlockCentroids:
    mu: torch.Tensor
    log_sigma: torch.Tensor
    weight_sum: torch.Tensor


@dataclass(frozen=True)
class CandidateSelection:
    indices: torch.Tensor
    mask: torch.Tensor
    superblock_budgets: torch.Tensor | None
    centroid_budgets: torch.Tensor
    token_budgets: torch.Tensor
    selected_superblock_mask: torch.Tensor | None
    opened_superblock_mask: torch.Tensor | None
    selected_centroid_mask: torch.Tensor
    opened_block_mask: torch.Tensor
    selected_shifted_superblock_mask: torch.Tensor | None = None
    opened_shifted_superblock_mask: torch.Tensor | None = None
    selected_shifted_centroid_mask: torch.Tensor | None = None
    opened_shifted_block_mask: torch.Tensor | None = None


def uncertainty_to_bucket_values(
    uncertainty: torch.Tensor,
    buckets: tuple[int, ...],
    *,
    beta: float,
    gamma: float,
    adaptive: bool = True,
) -> torch.Tensor:
    """Map scalar uncertainty values to monotonic bucket values."""

    if not adaptive:
        return torch.full_like(uncertainty, buckets[-1], dtype=torch.long)

    scaled = torch.sigmoid((uncertainty - beta) / gamma)
    bucket_count = len(buckets)
    bucket_index = torch.floor(scaled * bucket_count).long()
    bucket_index = bucket_index.clamp(min=0, max=bucket_count - 1)
    values = torch.tensor(buckets, device=uncertainty.device, dtype=torch.long)
    return values[bucket_index]


def summarize_block_centroids(
    mu_store: torch.Tensor,
    log_sigma_store: torch.Tensor,
    assignment_logits: torch.Tensor,
    *,
    block_size: int,
    token_importance: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> BlockCentroids:
    """Build four learned soft-assignment centroids per block.

    Shapes:
      mu_store: [batch, seq, routing_dim]
      log_sigma_store: [batch, seq, routing_dim]
      assignment_logits: [batch, seq, 4]
      token_importance: optional [batch, seq]
    """

    batch, seq_len, routing_dim = mu_store.shape
    centroids_per_block = assignment_logits.shape[-1]
    num_blocks = (seq_len + block_size - 1) // block_size
    padded_len = num_blocks * block_size
    pad_len = padded_len - seq_len

    if pad_len:
        mu_store = F.pad(mu_store, (0, 0, 0, pad_len))
        log_sigma_store = F.pad(log_sigma_store, (0, 0, 0, pad_len))
        assignment_logits = F.pad(assignment_logits, (0, 0, 0, pad_len))
        if token_importance is not None:
            token_importance = F.pad(token_importance, (0, pad_len))

    valid = torch.ones(batch, seq_len, 1, device=mu_store.device, dtype=mu_store.dtype)
    if pad_len:
        valid = F.pad(valid, (0, 0, 0, pad_len))
    if token_importance is not None:
        valid = valid * token_importance.unsqueeze(-1).to(valid.dtype)

    weights = assignment_logits.softmax(dim=-1) * valid
    weights = weights.view(batch, num_blocks, block_size, centroids_per_block)
    mu_blocks = mu_store.view(batch, num_blocks, block_size, routing_dim)
    log_sigma_blocks = log_sigma_store.view(batch, num_blocks, block_size, routing_dim)

    weights_expanded = weights.unsqueeze(-1)
    denom = weights_expanded.sum(dim=2).clamp_min(eps)
    mu = (weights_expanded * mu_blocks.unsqueeze(3)).sum(dim=2) / denom

    token_var = (2.0 * log_sigma_blocks).exp().unsqueeze(3)
    centered = mu_blocks.unsqueeze(3) - mu.unsqueeze(2)
    var = (weights_expanded * (token_var + centered.square())).sum(dim=2) / denom
    log_sigma = 0.5 * torch.log(var.clamp_min(eps))

    return BlockCentroids(mu=mu, log_sigma=log_sigma, weight_sum=denom.squeeze(-1))


def summarize_shifted_block_centroids(
    mu_store: torch.Tensor,
    log_sigma_store: torch.Tensor,
    assignment_logits: torch.Tensor,
    *,
    block_size: int,
    block_start_offset: int,
    token_importance: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> BlockCentroids:
    """Build block centroids for a second grid starting at ``block_start_offset``."""

    if block_start_offset <= 0:
        raise ValueError("block_start_offset must be positive")

    batch, seq_len, routing_dim = mu_store.shape
    centroids_per_block = assignment_logits.shape[-1]
    if block_start_offset >= seq_len:
        return BlockCentroids(
            mu=mu_store.new_zeros(batch, 0, centroids_per_block, routing_dim),
            log_sigma=log_sigma_store.new_zeros(batch, 0, centroids_per_block, routing_dim),
            weight_sum=mu_store.new_zeros(batch, 0, centroids_per_block),
        )

    shifted_importance = (
        token_importance[:, block_start_offset:] if token_importance is not None else None
    )
    return summarize_block_centroids(
        mu_store[:, block_start_offset:],
        log_sigma_store[:, block_start_offset:],
        assignment_logits[:, block_start_offset:],
        block_size=block_size,
        token_importance=shifted_importance,
        eps=eps,
    )


def summarize_superblock_centroids(
    block_centroids: BlockCentroids,
    *,
    superblock_size_blocks: int,
    eps: float = 1e-6,
) -> BlockCentroids:
    """Summarize groups of blocks into four superblock centroids."""

    batch, num_blocks, centroids_per_block, routing_dim = block_centroids.mu.shape
    num_superblocks = (num_blocks + superblock_size_blocks - 1) // superblock_size_blocks
    padded_blocks = num_superblocks * superblock_size_blocks
    pad_blocks = padded_blocks - num_blocks

    mu = block_centroids.mu
    log_sigma = block_centroids.log_sigma
    weights = block_centroids.weight_sum
    if pad_blocks:
        mu = torch.cat(
            [
                mu,
                torch.zeros(
                    batch,
                    pad_blocks,
                    centroids_per_block,
                    routing_dim,
                    device=mu.device,
                    dtype=mu.dtype,
                ),
            ],
            dim=1,
        )
        log_sigma = torch.cat(
            [
                log_sigma,
                torch.zeros(
                    batch,
                    pad_blocks,
                    centroids_per_block,
                    routing_dim,
                    device=log_sigma.device,
                    dtype=log_sigma.dtype,
                ),
            ],
            dim=1,
        )
        weights = torch.cat(
            [
                weights,
                torch.zeros(
                    batch,
                    pad_blocks,
                    centroids_per_block,
                    device=weights.device,
                    dtype=weights.dtype,
                ),
            ],
            dim=1,
        )

    mu = mu.view(batch, num_superblocks, superblock_size_blocks, centroids_per_block, routing_dim)
    log_sigma = log_sigma.view(
        batch,
        num_superblocks,
        superblock_size_blocks,
        centroids_per_block,
        routing_dim,
    )
    weights = weights.view(batch, num_superblocks, superblock_size_blocks, centroids_per_block)
    weights_expanded = weights.unsqueeze(-1)
    denom = weights_expanded.sum(dim=2).clamp_min(eps)
    super_mu = (weights_expanded * mu).sum(dim=2) / denom

    var = (2.0 * log_sigma).exp()
    centered = mu - super_mu.unsqueeze(2)
    super_var = (weights_expanded * (var + centered.square())).sum(dim=2) / denom
    super_log_sigma = 0.5 * torch.log(super_var.clamp_min(eps))

    return BlockCentroids(
        mu=super_mu,
        log_sigma=super_log_sigma,
        weight_sum=denom.squeeze(-1),
    )


def compute_centroid_route_scores(
    mu_seek: torch.Tensor,
    log_sigma_seek: torch.Tensor,
    centroids: BlockCentroids,
    config: UABLAConfig,
    *,
    block_start_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each token against routeable block centroids."""

    batch, seq_len, _ = mu_seek.shape
    num_blocks = centroids.mu.shape[1]
    distances = cheap_gaussian_distance(
        mu_seek[:, :, None, None, :],
        log_sigma_seek[:, :, None, None, :],
        centroids.mu[:, None, :, :, :],
        centroids.log_sigma[:, None, :, :, :],
        alpha=config.alpha,
        eps=config.eps,
    )
    route_scores = -distances / config.tau

    routeable_mask = _routeable_block_mask(
        batch=batch,
        seq_len=seq_len,
        num_blocks=num_blocks,
        centroids_per_block=config.centroids_per_block,
        block_size=config.block_size,
        block_start_offset=block_start_offset,
        device=mu_seek.device,
    )
    routeable_mask = routeable_mask.expand(batch, seq_len, num_blocks, config.centroids_per_block)
    route_scores = route_scores.masked_fill(
        ~routeable_mask,
        torch.finfo(route_scores.dtype).min,
    )
    return route_scores, routeable_mask


def compute_superblock_route_scores(
    mu_seek: torch.Tensor,
    log_sigma_seek: torch.Tensor,
    superblock_centroids: BlockCentroids,
    config: UABLAConfig,
    *,
    block_start_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each token against routeable superblock centroids."""

    batch, seq_len, _ = mu_seek.shape
    num_superblocks = superblock_centroids.mu.shape[1]
    distances = cheap_gaussian_distance(
        mu_seek[:, :, None, None, :],
        log_sigma_seek[:, :, None, None, :],
        superblock_centroids.mu[:, None, :, :, :],
        superblock_centroids.log_sigma[:, None, :, :, :],
        alpha=config.alpha,
        eps=config.eps,
    )
    route_scores = -distances / config.tau

    token_positions = torch.arange(seq_len, device=mu_seek.device)
    superblock_positions = torch.arange(num_superblocks, device=mu_seek.device)
    superblock_end_positions = block_start_offset + (
        (superblock_positions + 1) * config.superblock_size_blocks * config.block_size
    )
    routeable_mask = superblock_positions.view(
        1,
        1,
        num_superblocks,
        1,
    ) >= 0
    routeable_mask = routeable_mask & (
        superblock_end_positions.view(1, 1, num_superblocks, 1)
        <= (token_positions + 1).view(1, seq_len, 1, 1)
    )
    routeable_mask = routeable_mask.expand(
        batch,
        seq_len,
        num_superblocks,
        config.centroids_per_block,
    )
    route_scores = route_scores.masked_fill(
        ~routeable_mask,
        torch.finfo(route_scores.dtype).min,
    )
    return route_scores, routeable_mask


def build_candidate_indices(
    mu_seek: torch.Tensor,
    log_sigma_seek: torch.Tensor,
    centroids: BlockCentroids,
    config: UABLAConfig,
    *,
    route_scores: torch.Tensor | None = None,
    routeable_mask: torch.Tensor | None = None,
    superblock_route_scores: torch.Tensor | None = None,
    superblock_routeable_mask: torch.Tensor | None = None,
    shifted_route_scores: torch.Tensor | None = None,
    shifted_routeable_mask: torch.Tensor | None = None,
    shifted_superblock_route_scores: torch.Tensor | None = None,
    shifted_superblock_routeable_mask: torch.Tensor | None = None,
) -> CandidateSelection:
    """Build padded causal candidate indices without constructing a T x T matrix."""

    batch, seq_len, _ = mu_seek.shape
    device = mu_seek.device
    uncertainty = log_sigma_seek.mean(dim=-1)
    superblock_budgets = None
    if config.use_multiscale_routing:
        superblock_budgets = uncertainty_to_bucket_values(
            uncertainty,
            config.superblock_hit_buckets,
            beta=config.beta,
            gamma=config.gamma,
            adaptive=config.adaptive_budgets,
        )
    centroid_budgets = uncertainty_to_bucket_values(
        uncertainty,
        config.centroid_hit_buckets,
        beta=config.beta,
        gamma=config.gamma,
        adaptive=config.adaptive_budgets,
    )
    token_budgets = uncertainty_to_bucket_values(
        uncertainty,
        config.token_k_buckets,
        beta=config.beta,
        gamma=config.gamma,
        adaptive=config.adaptive_budgets,
    )

    max_candidates = config.local_window + max(config.centroid_hit_buckets) * config.block_size
    indices = torch.full((batch, seq_len, max_candidates), -1, device=device, dtype=torch.long)
    mask = torch.zeros((batch, seq_len, max_candidates), device=device, dtype=torch.bool)
    num_blocks = centroids.mu.shape[1]
    selected_centroid_mask = torch.zeros(
        (batch, seq_len, num_blocks, config.centroids_per_block),
        device=device,
        dtype=torch.bool,
    )
    opened_block_mask = torch.zeros((batch, seq_len, num_blocks), device=device, dtype=torch.bool)

    use_superblocks = (
        config.use_multiscale_routing
        and superblock_route_scores is not None
        and superblock_routeable_mask is not None
        and superblock_budgets is not None
    )
    selected_superblock_mask = None
    opened_superblock_mask = None
    if use_superblocks:
        num_superblocks = superblock_route_scores.shape[2]
        selected_superblock_mask = torch.zeros(
            (batch, seq_len, num_superblocks, config.centroids_per_block),
            device=device,
            dtype=torch.bool,
        )
        opened_superblock_mask = torch.zeros(
            (batch, seq_len, num_superblocks),
            device=device,
            dtype=torch.bool,
        )

    use_shifted = (
        config.use_shifted_blocks
        and shifted_route_scores is not None
        and shifted_routeable_mask is not None
        and shifted_route_scores.shape[2] > 0
    )
    shifted_offset = config.shifted_block_offset_value
    shifted_num_blocks = shifted_route_scores.shape[2] if use_shifted else 0
    selected_shifted_centroid_mask = None
    opened_shifted_block_mask = None
    selected_shifted_superblock_mask = None
    opened_shifted_superblock_mask = None
    if use_shifted:
        selected_shifted_centroid_mask = torch.zeros_like(shifted_routeable_mask)
        opened_shifted_block_mask = torch.zeros(
            (batch, seq_len, shifted_num_blocks),
            device=device,
            dtype=torch.bool,
        )
        if (
            config.use_multiscale_routing
            and shifted_superblock_route_scores is not None
            and shifted_superblock_routeable_mask is not None
            and superblock_budgets is not None
        ):
            selected_shifted_superblock_mask = torch.zeros_like(shifted_superblock_routeable_mask)
            opened_shifted_superblock_mask = torch.zeros(
                shifted_superblock_route_scores.shape[:3],
                device=device,
                dtype=torch.bool,
            )

    if route_scores is None or routeable_mask is None:
        route_scores, routeable_mask = compute_centroid_route_scores(
            mu_seek,
            log_sigma_seek,
            centroids,
            config,
        )
    route_scores_for_topk = route_scores.detach()
    superblock_scores_for_topk = (
        superblock_route_scores.detach() if superblock_route_scores is not None else None
    )

    with torch.no_grad():
        for batch_idx in range(batch):
            for token_idx in range(seq_len):
                seen: set[int] = set()
                out: list[int] = []

                local_start = max(0, token_idx - config.local_window + 1)
                for candidate_idx in range(local_start, token_idx + 1):
                    seen.add(candidate_idx)
                    out.append(candidate_idx)

                complete_blocks = min(num_blocks, (token_idx + 1) // config.block_size)
                block_pool = _select_block_pool(
                    batch_idx=batch_idx,
                    token_idx=token_idx,
                    complete_blocks=complete_blocks,
                    config=config,
                    superblock_budgets=superblock_budgets,
                    superblock_scores_for_topk=superblock_scores_for_topk,
                    superblock_routeable_mask=superblock_routeable_mask,
                    selected_superblock_mask=selected_superblock_mask,
                    opened_superblock_mask=opened_superblock_mask,
                )
                shifted_block_pool: list[int] = []
                if use_shifted:
                    shifted_complete_blocks = min(
                        shifted_num_blocks,
                        _complete_blocks_at_token(
                            token_idx,
                            block_size=config.block_size,
                            block_start_offset=shifted_offset,
                        ),
                    )
                    shifted_block_pool = _select_block_pool(
                        batch_idx=batch_idx,
                        token_idx=token_idx,
                        complete_blocks=shifted_complete_blocks,
                        config=config,
                        superblock_budgets=superblock_budgets,
                        superblock_scores_for_topk=(
                            shifted_superblock_route_scores.detach()
                            if shifted_superblock_route_scores is not None
                            else None
                        ),
                        superblock_routeable_mask=shifted_superblock_routeable_mask,
                        selected_superblock_mask=selected_shifted_superblock_mask,
                        opened_superblock_mask=opened_shifted_superblock_mask,
                    )

                centroid_budget = int(centroid_budgets[batch_idx, token_idx].item())
                if centroid_budget > 0 and (block_pool or shifted_block_pool):
                    score_parts: list[torch.Tensor] = []
                    routeable_parts: list[torch.Tensor] = []
                    block_tensors: list[torch.Tensor | None] = [None, None]
                    if complete_blocks > 0 and block_pool:
                        block_tensor = torch.tensor(block_pool, device=device, dtype=torch.long)
                        block_tensors[0] = block_tensor
                        block_scores = route_scores_for_topk[batch_idx, token_idx, block_tensor]
                        block_routeable = routeable_mask[batch_idx, token_idx, block_tensor]
                        score_parts.append(
                            block_scores.reshape(block_tensor.numel() * config.centroids_per_block),
                        )
                        routeable_parts.append(
                            block_routeable.reshape(
                                block_tensor.numel() * config.centroids_per_block,
                            ),
                        )
                    if use_shifted and shifted_block_pool:
                        shifted_block_tensor = torch.tensor(
                            shifted_block_pool,
                            device=device,
                            dtype=torch.long,
                        )
                        block_tensors[1] = shifted_block_tensor
                        shifted_block_scores = shifted_route_scores.detach()[
                            batch_idx,
                            token_idx,
                            shifted_block_tensor,
                        ]
                        shifted_block_routeable = shifted_routeable_mask[
                            batch_idx,
                            token_idx,
                            shifted_block_tensor,
                        ]
                        score_parts.append(
                            shifted_block_scores.reshape(
                                shifted_block_tensor.numel() * config.centroids_per_block,
                            ),
                        )
                        routeable_parts.append(
                            shifted_block_routeable.reshape(
                                shifted_block_tensor.numel() * config.centroids_per_block,
                            ),
                        )

                    flat_scores = torch.cat(score_parts, dim=0)
                    flat_routeable = torch.cat(routeable_parts, dim=0)
                    valid_count = int(flat_routeable.sum().item())
                    if valid_count > 0:
                        top_k = min(centroid_budget, valid_count)
                        flat_scores = flat_scores.masked_fill(
                            ~flat_routeable,
                            torch.finfo(flat_scores.dtype).min,
                        )
                        top_centroids = torch.topk(flat_scores, k=top_k).indices

                        opened_blocks: list[int] = []
                        opened_seen: set[int] = set()
                        opened_shifted_blocks: list[int] = []
                        opened_shifted_seen: set[int] = set()
                        base_centroid_count = (
                            0
                            if block_tensors[0] is None
                            else block_tensors[0].numel() * config.centroids_per_block
                        )
                        for centroid_idx in top_centroids.tolist():
                            if centroid_idx < base_centroid_count:
                                block_tensor = block_tensors[0]
                                if block_tensor is None:
                                    continue
                                pool_idx = centroid_idx // config.centroids_per_block
                                block_idx = int(block_tensor[pool_idx].item())
                                local_centroid_idx = centroid_idx % config.centroids_per_block
                                selected_centroid_mask[
                                    batch_idx,
                                    token_idx,
                                    block_idx,
                                    local_centroid_idx,
                                ] = True
                                if block_idx not in opened_seen:
                                    opened_seen.add(block_idx)
                                    opened_blocks.append(block_idx)
                                    opened_block_mask[batch_idx, token_idx, block_idx] = True
                            elif use_shifted and selected_shifted_centroid_mask is not None:
                                shifted_block_tensor = block_tensors[1]
                                if shifted_block_tensor is None:
                                    continue
                                shifted_centroid_idx = centroid_idx - base_centroid_count
                                pool_idx = shifted_centroid_idx // config.centroids_per_block
                                block_idx = int(shifted_block_tensor[pool_idx].item())
                                local_centroid_idx = (
                                    shifted_centroid_idx % config.centroids_per_block
                                )
                                selected_shifted_centroid_mask[
                                    batch_idx,
                                    token_idx,
                                    block_idx,
                                    local_centroid_idx,
                                ] = True
                                if block_idx not in opened_shifted_seen:
                                    opened_shifted_seen.add(block_idx)
                                    opened_shifted_blocks.append(block_idx)
                                    if opened_shifted_block_mask is not None:
                                        opened_shifted_block_mask[
                                            batch_idx,
                                            token_idx,
                                            block_idx,
                                        ] = True

                        for block_idx in opened_blocks:
                            block_start = block_idx * config.block_size
                            block_end = min(block_start + config.block_size, token_idx + 1, seq_len)
                            for candidate_idx in range(block_start, block_end):
                                if candidate_idx not in seen:
                                    seen.add(candidate_idx)
                                    out.append(candidate_idx)
                        for block_idx in opened_shifted_blocks:
                            block_start = shifted_offset + block_idx * config.block_size
                            block_end = min(block_start + config.block_size, token_idx + 1, seq_len)
                            for candidate_idx in range(block_start, block_end):
                                if candidate_idx not in seen:
                                    seen.add(candidate_idx)
                                    out.append(candidate_idx)

                capped = out[:max_candidates]
                if capped:
                    values = torch.tensor(capped, device=device, dtype=torch.long)
                    count = values.numel()
                    indices[batch_idx, token_idx, :count] = values
                    mask[batch_idx, token_idx, :count] = True

    return CandidateSelection(
        indices=indices,
        mask=mask,
        superblock_budgets=superblock_budgets,
        centroid_budgets=centroid_budgets,
        token_budgets=token_budgets,
        selected_superblock_mask=selected_superblock_mask,
        opened_superblock_mask=opened_superblock_mask,
        selected_centroid_mask=selected_centroid_mask,
        opened_block_mask=opened_block_mask,
        selected_shifted_superblock_mask=selected_shifted_superblock_mask,
        opened_shifted_superblock_mask=opened_shifted_superblock_mask,
        selected_shifted_centroid_mask=selected_shifted_centroid_mask,
        opened_shifted_block_mask=opened_shifted_block_mask,
    )


def build_candidate_indices_vectorized(
    mu_seek: torch.Tensor,
    log_sigma_seek: torch.Tensor,
    centroids: BlockCentroids,
    config: UABLAConfig,
    *,
    route_scores: torch.Tensor,
    routeable_mask: torch.Tensor,
    superblock_route_scores: torch.Tensor | None = None,
    superblock_routeable_mask: torch.Tensor | None = None,
    shifted_route_scores: torch.Tensor | None = None,
    shifted_routeable_mask: torch.Tensor | None = None,
    shifted_superblock_route_scores: torch.Tensor | None = None,
    shifted_superblock_routeable_mask: torch.Tensor | None = None,
) -> CandidateSelection:
    """Vectorized fixed-shape candidate builder for GPU/MPS-friendly training."""

    batch, seq_len, _ = mu_seek.shape
    device = mu_seek.device
    num_blocks = centroids.mu.shape[1]
    uncertainty = log_sigma_seek.mean(dim=-1)
    superblock_budgets = None
    if config.use_multiscale_routing:
        superblock_budgets = uncertainty_to_bucket_values(
            uncertainty,
            config.superblock_hit_buckets,
            beta=config.beta,
            gamma=config.gamma,
            adaptive=config.adaptive_budgets,
        )
    centroid_budgets = uncertainty_to_bucket_values(
        uncertainty,
        config.centroid_hit_buckets,
        beta=config.beta,
        gamma=config.gamma,
        adaptive=config.adaptive_budgets,
    )
    token_budgets = uncertainty_to_bucket_values(
        uncertainty,
        config.token_k_buckets,
        beta=config.beta,
        gamma=config.gamma,
        adaptive=config.adaptive_budgets,
    )

    selected_superblock_mask = None
    opened_superblock_mask = None
    superblock_block_mask = None
    if (
        config.use_multiscale_routing
        and superblock_route_scores is not None
        and superblock_routeable_mask is not None
    ):
        selected_superblock_mask, opened_superblock_mask, superblock_block_mask = (
            _superblock_to_block_mask(
                superblock_route_scores.detach(),
                superblock_routeable_mask,
                superblock_budgets,
                config,
                num_blocks=num_blocks,
                seq_len=seq_len,
                block_start_offset=0,
            )
        )

    constrained_routeable = routeable_mask
    if superblock_block_mask is not None:
        constrained_routeable = routeable_mask & superblock_block_mask.unsqueeze(-1)

    use_shifted = (
        config.use_shifted_blocks
        and shifted_route_scores is not None
        and shifted_routeable_mask is not None
        and shifted_route_scores.shape[2] > 0
    )
    shifted_offset = config.shifted_block_offset_value
    shifted_num_blocks = shifted_route_scores.shape[2] if use_shifted else 0
    selected_shifted_superblock_mask = None
    opened_shifted_superblock_mask = None
    shifted_superblock_block_mask = None
    if (
        use_shifted
        and config.use_multiscale_routing
        and shifted_superblock_route_scores is not None
        and shifted_superblock_routeable_mask is not None
    ):
        (
            selected_shifted_superblock_mask,
            opened_shifted_superblock_mask,
            shifted_superblock_block_mask,
        ) = _superblock_to_block_mask(
            shifted_superblock_route_scores.detach(),
            shifted_superblock_routeable_mask,
            superblock_budgets,
            config,
            num_blocks=shifted_num_blocks,
            seq_len=seq_len,
            block_start_offset=shifted_offset,
        )

    shifted_constrained_routeable = None
    if use_shifted:
        shifted_constrained_routeable = shifted_routeable_mask
        if shifted_superblock_block_mask is not None:
            shifted_constrained_routeable = (
                shifted_routeable_mask & shifted_superblock_block_mask.unsqueeze(-1)
            )

    flat_scores = route_scores.detach().reshape(batch, seq_len, -1)
    flat_valid = constrained_routeable.reshape(batch, seq_len, -1)
    base_flat_count = flat_scores.shape[-1]
    if use_shifted and shifted_constrained_routeable is not None:
        flat_scores = torch.cat(
            [flat_scores, shifted_route_scores.detach().reshape(batch, seq_len, -1)],
            dim=-1,
        )
        flat_valid = torch.cat(
            [flat_valid, shifted_constrained_routeable.reshape(batch, seq_len, -1)],
            dim=-1,
        )

    selected_flat, top_centroid_indices, top_centroid_valid = _topk_flat_per_token(
        flat_scores,
        flat_valid,
        centroid_budgets,
        max(config.centroid_hit_buckets),
    )
    selected_centroid_mask = selected_flat[..., :base_flat_count].reshape_as(routeable_mask)
    opened_block_mask = selected_centroid_mask.any(dim=-1)
    selected_shifted_centroid_mask = None
    opened_shifted_block_mask = None
    if use_shifted and shifted_routeable_mask is not None:
        selected_shifted_centroid_mask = selected_flat[..., base_flat_count:].reshape_as(
            shifted_routeable_mask,
        )
        opened_shifted_block_mask = selected_shifted_centroid_mask.any(dim=-1)

    top_is_shifted = top_centroid_indices >= base_flat_count
    base_top_block_indices = top_centroid_indices // config.centroids_per_block
    shifted_top_block_indices = (
        (top_centroid_indices - base_flat_count).clamp_min(0) // config.centroids_per_block
    )
    top_block_starts = base_top_block_indices * config.block_size
    if use_shifted:
        shifted_block_starts = shifted_offset + shifted_top_block_indices * config.block_size
        top_block_starts = torch.where(top_is_shifted, shifted_block_starts, top_block_starts)
    top_block_valid = top_centroid_valid

    local_indices, local_mask = _local_candidate_indices(
        batch=batch,
        seq_len=seq_len,
        local_window=config.local_window,
        device=device,
    )
    routed_indices, routed_mask = _routed_block_candidate_indices_from_starts(
        top_block_starts,
        top_block_valid,
        seq_len=seq_len,
        block_size=config.block_size,
    )

    indices = torch.cat([local_indices, routed_indices], dim=-1)
    mask = torch.cat([local_mask, routed_mask], dim=-1)
    indices, mask = _mask_duplicate_candidates(indices, mask)

    return CandidateSelection(
        indices=indices,
        mask=mask,
        superblock_budgets=superblock_budgets,
        centroid_budgets=centroid_budgets,
        token_budgets=token_budgets,
        selected_superblock_mask=selected_superblock_mask,
        opened_superblock_mask=opened_superblock_mask,
        selected_centroid_mask=selected_centroid_mask,
        opened_block_mask=opened_block_mask,
        selected_shifted_superblock_mask=selected_shifted_superblock_mask,
        opened_shifted_superblock_mask=opened_shifted_superblock_mask,
        selected_shifted_centroid_mask=selected_shifted_centroid_mask,
        opened_shifted_block_mask=opened_shifted_block_mask,
    )


def _topk_flat_per_token(
    flat_scores: torch.Tensor,
    flat_valid: torch.Tensor,
    budgets: torch.Tensor,
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_scores = flat_scores.masked_fill(~flat_valid, torch.finfo(flat_scores.dtype).min)
    max_k = min(max_k, flat_scores.shape[-1])
    top_idx = torch.topk(safe_scores, k=max_k, dim=-1).indices
    top_valid = torch.gather(flat_valid, dim=-1, index=top_idx)
    budget_mask = (
        torch.arange(max_k, device=flat_scores.device).view(1, 1, max_k)
        < budgets.clamp(max=max_k).unsqueeze(-1)
    )
    top_valid = top_valid & budget_mask
    selected = torch.zeros_like(flat_valid)
    selected.scatter_(dim=-1, index=top_idx, src=top_valid)
    selected = selected & flat_valid
    return selected, top_idx, top_valid


def _topk_per_token(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    budgets: torch.Tensor,
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_scores = scores.reshape(*scores.shape[:2], -1)
    flat_valid = valid_mask.reshape(*valid_mask.shape[:2], -1)
    selected, top_idx, top_valid = _topk_flat_per_token(
        flat_scores,
        flat_valid,
        budgets,
        max_k,
    )
    return selected.reshape_as(valid_mask), top_idx, top_valid


def _local_candidate_indices(
    *,
    batch: int,
    seq_len: int,
    local_window: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, device=device)
    offsets = torch.arange(local_window - 1, -1, -1, device=device)
    indices = positions.view(seq_len, 1) - offsets.view(1, local_window)
    mask = indices >= 0
    indices = indices.clamp_min(0)
    return (
        indices.view(1, seq_len, local_window).expand(batch, seq_len, local_window),
        mask.view(1, seq_len, local_window).expand(batch, seq_len, local_window),
    )


def _routed_block_candidate_indices(
    block_indices: torch.Tensor,
    block_valid: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _routed_block_candidate_indices_from_starts(
        block_indices * block_size,
        block_valid,
        seq_len=seq_len,
        block_size=block_size,
    )


def _routed_block_candidate_indices_from_starts(
    block_starts: torch.Tensor,
    block_valid: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len_from_blocks, block_count = block_starts.shape
    device = block_starts.device
    offsets = torch.arange(block_size, device=device)
    block_token_indices = block_starts.unsqueeze(-1) + offsets.view(1, 1, 1, block_size)
    indices = block_token_indices.reshape(batch, seq_len_from_blocks, block_count * block_size)
    token_positions = torch.arange(seq_len, device=device)
    block_token_mask = indices < seq_len
    opened_token_mask = block_valid.unsqueeze(-1).expand(
        batch,
        seq_len_from_blocks,
        block_count,
        block_size,
    ).reshape(batch, seq_len_from_blocks, block_count * block_size)
    causal_mask = indices <= token_positions.view(1, seq_len, 1)
    mask = opened_token_mask & causal_mask & block_token_mask
    return indices.clamp(max=max(seq_len - 1, 0)), mask


def _expand_candidate_spans(
    indices: torch.Tensor,
    mask: torch.Tensor,
    *,
    seq_len: int,
    left: int,
    right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if left < 0 or right < 0:
        raise ValueError("span expansion values cannot be negative")
    if left == 0 and right == 0:
        return indices, mask
    device = indices.device
    offsets = torch.arange(-left, right + 1, device=device)
    expanded = indices.unsqueeze(-1) + offsets.view(*((1,) * indices.ndim), offsets.numel())
    expanded = expanded.reshape(*indices.shape[:-1], indices.shape[-1] * offsets.numel())
    expanded_mask = mask.unsqueeze(-1).expand(
        *mask.shape,
        offsets.numel(),
    ).reshape_as(expanded)
    query_len = indices.shape[1]
    token_positions = torch.arange(query_len, device=device).view(1, query_len, 1)
    expanded_mask = expanded_mask & (expanded >= 0) & (expanded < seq_len) & (expanded <= token_positions)
    return expanded.clamp(min=0, max=max(seq_len - 1, 0)), expanded_mask


def _mask_duplicate_candidates(
    indices: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_count = indices.shape[-1]
    order = torch.arange(candidate_count, device=indices.device).view(
        *((1,) * (indices.ndim - 1)),
        candidate_count,
    )
    safe_indices = indices.clamp_min(0)
    invalid_index = safe_indices.amax(dim=-1, keepdim=True) + 1
    safe_indices = torch.where(mask, safe_indices, invalid_index)
    sort_key = safe_indices * (candidate_count + 1) + order
    _, sorted_positions = sort_key.sort(dim=-1)
    sorted_indices = torch.gather(indices, dim=-1, index=sorted_positions)
    sorted_mask = torch.gather(mask, dim=-1, index=sorted_positions)

    previous_same = sorted_indices[..., 1:] == sorted_indices[..., :-1]
    previous_valid = sorted_mask[..., 1:] & sorted_mask[..., :-1]
    duplicate_sorted = torch.zeros_like(sorted_mask)
    duplicate_sorted[..., 1:] = previous_same & previous_valid

    duplicate = torch.zeros_like(mask)
    duplicate.scatter_(dim=-1, index=sorted_positions, src=duplicate_sorted)
    return indices, mask & ~duplicate


def _routeable_block_mask(
    *,
    batch: int,
    seq_len: int,
    num_blocks: int,
    centroids_per_block: int,
    block_size: int,
    block_start_offset: int,
    device: torch.device,
) -> torch.Tensor:
    token_positions = torch.arange(seq_len, device=device)
    block_positions = torch.arange(num_blocks, device=device)
    block_end_positions = block_start_offset + (block_positions + 1) * block_size
    routeable_mask = block_end_positions.view(1, 1, num_blocks, 1) <= (
        token_positions + 1
    ).view(1, seq_len, 1, 1)
    return routeable_mask.expand(batch, seq_len, num_blocks, centroids_per_block)


def _complete_blocks_by_position(
    *,
    seq_len: int,
    block_size: int,
    block_start_offset: int,
    device: torch.device,
) -> torch.Tensor:
    token_positions = torch.arange(seq_len, device=device)
    complete = torch.div(
        token_positions + 1 - block_start_offset,
        block_size,
        rounding_mode="floor",
    )
    return complete.clamp_min(0)


def _complete_blocks_at_token(
    token_idx: int,
    *,
    block_size: int,
    block_start_offset: int,
) -> int:
    return max(0, (token_idx + 1 - block_start_offset) // block_size)


def _superblock_to_block_mask(
    superblock_route_scores: torch.Tensor,
    superblock_routeable_mask: torch.Tensor,
    superblock_budgets: torch.Tensor | None,
    config: UABLAConfig,
    *,
    num_blocks: int,
    seq_len: int,
    block_start_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if superblock_budgets is None:
        raise ValueError("superblock_budgets are required for superblock routing")

    selected_superblock_mask, _, _ = _topk_per_token(
        superblock_route_scores,
        superblock_routeable_mask,
        superblock_budgets,
        max(config.superblock_hit_buckets),
    )
    opened_superblock_mask = selected_superblock_mask.any(dim=-1)
    batch = opened_superblock_mask.shape[0]
    num_superblocks = opened_superblock_mask.shape[-1]
    device = superblock_route_scores.device

    block_ids = torch.arange(num_blocks, device=device)
    superblock_ids = (block_ids // config.superblock_size_blocks).clamp(max=num_superblocks - 1)
    superblock_block_mask = torch.gather(
        opened_superblock_mask,
        dim=2,
        index=superblock_ids.view(1, 1, num_blocks).expand(batch, seq_len, num_blocks),
    )

    complete_blocks = _complete_blocks_by_position(
        seq_len=seq_len,
        block_size=config.block_size,
        block_start_offset=block_start_offset,
        device=device,
    ).clamp(max=num_blocks)
    block_positions = torch.arange(num_blocks, device=device)
    tail_start = (complete_blocks // config.superblock_size_blocks) * config.superblock_size_blocks
    tail_mask = (
        (block_positions.view(1, num_blocks) >= tail_start.view(seq_len, 1))
        & (block_positions.view(1, num_blocks) < complete_blocks.view(seq_len, 1))
    )
    superblock_block_mask = superblock_block_mask | tail_mask.view(1, seq_len, num_blocks)
    return selected_superblock_mask, opened_superblock_mask, superblock_block_mask


def _select_block_pool(
    *,
    batch_idx: int,
    token_idx: int,
    complete_blocks: int,
    config: UABLAConfig,
    superblock_budgets: torch.Tensor | None,
    superblock_scores_for_topk: torch.Tensor | None,
    superblock_routeable_mask: torch.Tensor | None,
    selected_superblock_mask: torch.Tensor | None,
    opened_superblock_mask: torch.Tensor | None,
) -> list[int]:
    if complete_blocks <= 0:
        return []
    if (
        not config.use_multiscale_routing
        or superblock_budgets is None
        or superblock_scores_for_topk is None
        or superblock_routeable_mask is None
        or selected_superblock_mask is None
        or opened_superblock_mask is None
    ):
        return list(range(complete_blocks))

    complete_superblocks = complete_blocks // config.superblock_size_blocks
    if complete_superblocks <= 0:
        return list(range(complete_blocks))

    flat_scores = superblock_scores_for_topk[
        batch_idx,
        token_idx,
        :complete_superblocks,
    ].reshape(complete_superblocks * config.centroids_per_block)
    flat_routeable = superblock_routeable_mask[
        batch_idx,
        token_idx,
        :complete_superblocks,
    ].reshape(complete_superblocks * config.centroids_per_block)
    valid_count = int(flat_routeable.sum().item())
    if valid_count <= 0:
        return list(range(complete_blocks))

    superblock_budget = int(superblock_budgets[batch_idx, token_idx].item())
    top_k = min(superblock_budget, valid_count)
    flat_scores = flat_scores.masked_fill(~flat_routeable, torch.finfo(flat_scores.dtype).min)
    top_supercentroids = torch.topk(flat_scores, k=top_k).indices

    opened_superblocks: list[int] = []
    opened_seen: set[int] = set()
    for supercentroid_idx in top_supercentroids.tolist():
        superblock_idx = supercentroid_idx // config.centroids_per_block
        local_centroid_idx = supercentroid_idx % config.centroids_per_block
        selected_superblock_mask[
            batch_idx,
            token_idx,
            superblock_idx,
            local_centroid_idx,
        ] = True
        if superblock_idx not in opened_seen:
            opened_seen.add(superblock_idx)
            opened_superblocks.append(superblock_idx)
            opened_superblock_mask[batch_idx, token_idx, superblock_idx] = True

    block_pool: set[int] = set()
    for superblock_idx in opened_superblocks:
        block_start = superblock_idx * config.superblock_size_blocks
        block_end = min(block_start + config.superblock_size_blocks, complete_blocks)
        block_pool.update(range(block_start, block_end))

    tail_start = complete_superblocks * config.superblock_size_blocks
    block_pool.update(range(tail_start, complete_blocks))
    return sorted(block_pool)
