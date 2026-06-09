"""PyTorch implementation of the locked UABLA-448 V1 attention module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import UABLAConfig
from .distance import cheap_gaussian_distance
from .routing import (
    BlockCentroids,
    CandidateSelection,
    build_candidate_indices,
    build_candidate_indices_vectorized,
    compute_centroid_route_scores,
    compute_superblock_route_scores,
    summarize_block_centroids,
    summarize_superblock_centroids,
)


@dataclass(frozen=True)
class UABLAOutput:
    output: torch.Tensor
    cache: torch.Tensor | None = None
    centroids: BlockCentroids | None = None
    superblock_centroids: BlockCentroids | None = None
    candidates: CandidateSelection | None = None
    route_scores: torch.Tensor | None = None
    routeable_mask: torch.Tensor | None = None
    superblock_route_scores: torch.Tensor | None = None
    superblock_routeable_mask: torch.Tensor | None = None
    attention: torch.Tensor | None = None
    token_scores: torch.Tensor | None = None
    memory_importance: torch.Tensor | None = None


class UABLAAttention(nn.Module):
    """Uncertainty-Adaptive Block Latent Attention.

    This is a clean reference implementation, not a fused production kernel.
    """

    def __init__(self, config: UABLAConfig) -> None:
        super().__init__()
        self.config = config

        self.q_mu = nn.Linear(config.hidden_size, config.routing_dim)
        self.q_log_sigma = nn.Linear(config.hidden_size, config.routing_dim)
        self.s_mu = nn.Linear(config.hidden_size, config.routing_dim)
        self.s_log_sigma = nn.Linear(config.hidden_size, config.routing_dim)
        self.v_down = nn.Linear(config.hidden_size, config.value_dim)
        self.pos_down = nn.Linear(config.hidden_size, config.position_dim)
        self.assign = nn.Linear(config.routing_dim, config.centroids_per_block)
        self.memory_importance = (
            nn.Linear(config.hidden_size, 1) if config.use_memory_importance else None
        )
        if config.hybrid_score_rank > 0 and config.hybrid_score_scale > 0:
            self.hybrid_q = nn.Linear(config.routing_dim, config.hybrid_score_rank, bias=False)
            self.hybrid_s = nn.Linear(config.routing_dim, config.hybrid_score_rank, bias=False)
        else:
            self.hybrid_q = None
            self.hybrid_s = None
        self.v_up = nn.Linear(config.value_dim, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.rel_pos_bias = nn.Embedding(config.max_position_buckets, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_cache: bool = False,
        return_routing: bool = False,
        return_attention: bool = False,
        return_token_scores: bool = False,
    ) -> torch.Tensor | UABLAOutput:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq, hidden]")

        config = self.config
        batch, seq_len, hidden = x.shape
        if hidden != config.hidden_size:
            raise ValueError(f"expected hidden size {config.hidden_size}, got {hidden}")

        mu_seek = self.q_mu(x)
        log_sigma_seek = self.q_log_sigma(x).clamp(
            min=config.min_log_sigma,
            max=config.max_log_sigma,
        )
        mu_store = self.s_mu(x)
        log_sigma_store = self.s_log_sigma(x).clamp(
            min=config.min_log_sigma,
            max=config.max_log_sigma,
        )
        compressed_value = self.v_down(x)
        position = self.pos_down(x)
        assignment_logits = self.assign(mu_store)
        memory_logit = None
        memory_importance = None
        if self.memory_importance is not None:
            memory_logit = self.memory_importance(x).squeeze(-1)
            memory_importance = torch.sigmoid(memory_logit)
            position = position.clone()
            position[..., 0] = memory_logit

        centroids = summarize_block_centroids(
            mu_store,
            log_sigma_store,
            assignment_logits,
            block_size=config.block_size,
            token_importance=memory_importance,
            eps=config.eps,
        )
        superblock_centroids = None
        superblock_route_scores = None
        superblock_routeable_mask = None
        if config.use_multiscale_routing:
            superblock_centroids = summarize_superblock_centroids(
                centroids,
                superblock_size_blocks=config.superblock_size_blocks,
                eps=config.eps,
            )
            superblock_route_scores, superblock_routeable_mask = compute_superblock_route_scores(
                mu_seek,
                log_sigma_seek,
                superblock_centroids,
                config,
            )
        route_scores, routeable_mask = compute_centroid_route_scores(
            mu_seek,
            log_sigma_seek,
            centroids,
            config,
        )
        if config.vectorized_routing:
            candidates = build_candidate_indices_vectorized(
                mu_seek,
                log_sigma_seek,
                centroids,
                config,
                route_scores=route_scores,
                routeable_mask=routeable_mask,
                superblock_route_scores=superblock_route_scores,
                superblock_routeable_mask=superblock_routeable_mask,
            )
        else:
            candidates = build_candidate_indices(
                mu_seek,
                log_sigma_seek,
                centroids,
                config,
                route_scores=route_scores,
                routeable_mask=routeable_mask,
                superblock_route_scores=superblock_route_scores,
                superblock_routeable_mask=superblock_routeable_mask,
            )

        safe_indices = candidates.indices.clamp_min(0)
        batch_indices = torch.arange(batch, device=x.device).view(batch, 1, 1)
        candidate_mu = mu_store[batch_indices, safe_indices]
        candidate_log_sigma = log_sigma_store[batch_indices, safe_indices]
        candidate_values = compressed_value[batch_indices, safe_indices]
        candidate_importance = (
            memory_importance[batch_indices, safe_indices] if memory_importance is not None else None
        )

        scores = -cheap_gaussian_distance(
            mu_seek.unsqueeze(2),
            log_sigma_seek.unsqueeze(2),
            candidate_mu,
            candidate_log_sigma,
            alpha=config.alpha,
            eps=config.eps,
        ) / config.tau
        if self.hybrid_q is not None and self.hybrid_s is not None:
            scores = scores + self._hybrid_score(mu_seek, candidate_mu)
        if candidate_importance is not None and config.memory_importance_score_scale > 0:
            scores = scores + config.memory_importance_score_scale * torch.log(
                candidate_importance.clamp_min(config.eps),
            )
        scores = scores + self._candidate_block_prior(
            candidates.indices,
            candidates.mask,
            route_scores,
            routeable_mask,
            candidates.opened_block_mask,
        )
        scores = scores + self._relative_position_bias(candidates.indices, seq_len)
        scores = scores.masked_fill(~candidates.mask, torch.finfo(scores.dtype).min)

        max_token_k = min(max(config.token_k_buckets), scores.shape[-1])
        _, top_positions = torch.topk(scores, k=max_token_k, dim=-1)
        rank = torch.arange(max_token_k, device=x.device).view(1, 1, max_token_k)
        token_budget_mask = rank < candidates.token_budgets.unsqueeze(-1)
        hard_token_mask = torch.zeros_like(candidates.mask)
        hard_token_mask.scatter_(dim=2, index=top_positions, src=token_budget_mask)
        hard_token_mask = hard_token_mask & candidates.mask

        hard_scores = scores.masked_fill(~hard_token_mask, torch.finfo(scores.dtype).min)
        attn_hard = torch.softmax(hard_scores, dim=-1)
        attn_soft = torch.softmax(scores, dim=-1)
        attn = attn_hard + attn_soft - attn_soft.detach()
        attn = self.dropout(attn)
        mixed_compressed = (attn.unsqueeze(-1) * candidate_values).sum(dim=2)
        mixed = F.linear(mixed_compressed, self.v_up.weight)
        if self.v_up.bias is not None:
            mixed = mixed + attn.sum(dim=-1, keepdim=True) * self.v_up.bias.view(1, 1, -1)
        output = self.out(mixed)

        cache = None
        if return_cache:
            cache = torch.cat([mu_store, log_sigma_store, compressed_value, position], dim=-1)

        if return_cache or return_routing or return_attention or return_token_scores:
            return UABLAOutput(
                output=output,
                cache=cache,
                centroids=centroids if return_routing else None,
                superblock_centroids=superblock_centroids if return_routing else None,
                candidates=candidates if return_routing else None,
                route_scores=route_scores if return_routing else None,
                routeable_mask=routeable_mask if return_routing else None,
                superblock_route_scores=superblock_route_scores if return_routing else None,
                superblock_routeable_mask=superblock_routeable_mask if return_routing else None,
                attention=attn if return_attention else None,
                token_scores=scores if return_attention or return_token_scores else None,
                memory_importance=memory_importance if return_routing else None,
            )
        return output

    def _hybrid_score(self, mu_seek: torch.Tensor, candidate_mu: torch.Tensor) -> torch.Tensor:
        if self.hybrid_q is None or self.hybrid_s is None:
            raise RuntimeError("hybrid score projections are not initialized")
        q = self.hybrid_q(mu_seek)
        s = self.hybrid_s(candidate_mu)
        scale = self.config.hybrid_score_scale / (self.config.hybrid_score_rank**0.5)
        return scale * (q.unsqueeze(2) * s).sum(dim=-1)

    def _candidate_block_prior(
        self,
        candidate_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
        route_scores: torch.Tensor,
        routeable_mask: torch.Tensor,
        opened_block_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, num_blocks, centroids_per_block = route_scores.shape
        flat_scores = route_scores.reshape(batch, seq_len, num_blocks * centroids_per_block)
        flat_routeable = routeable_mask.reshape(batch, seq_len, num_blocks * centroids_per_block)
        safe_scores = flat_scores.masked_fill(~flat_routeable, torch.finfo(flat_scores.dtype).min)
        soft_centroid = torch.softmax(safe_scores, dim=-1) * flat_routeable.to(flat_scores.dtype)
        soft_denominator = soft_centroid.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        soft_centroid = soft_centroid / soft_denominator
        soft_block = soft_centroid.reshape(batch, seq_len, num_blocks, centroids_per_block).sum(
            dim=-1,
        )

        hard_block = opened_block_mask.to(route_scores.dtype)
        straight_through_block = hard_block + soft_block - soft_block.detach()

        candidate_blocks = (candidate_indices.clamp_min(0) // self.config.block_size).clamp(
            max=num_blocks - 1,
        )
        gathered_prior = torch.gather(straight_through_block, dim=2, index=candidate_blocks)
        gathered_opened = torch.gather(opened_block_mask, dim=2, index=candidate_blocks)
        prior_bias = torch.log(gathered_prior.clamp_min(self.config.eps))
        return torch.where(gathered_opened & candidate_mask, prior_bias, torch.zeros_like(prior_bias))

    def _relative_position_bias(self, candidate_indices: torch.Tensor, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=candidate_indices.device).view(1, seq_len, 1)
        distances = (positions - candidate_indices.clamp_min(0)).clamp_min(0)
        buckets = distances.clamp_max(self.config.max_position_buckets - 1)
        return self.rel_pos_bias(buckets).squeeze(-1)
