"""Prototype hierarchical memory cache and revival routing.

This module is intentionally next to UABLA rather than inside the attention
implementation. It simulates an inference-time cache policy:

recent tokens stay exact, old tokens are summarized into leaves, and queries
revive exact old leaves only when summary routing selects them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None

    class _NoTorchNN:
        Module = object

    nn = _NoTorchNN()


@dataclass(frozen=True)
class RevivalCacheConfig:
    """Configuration for the first revival-cache simulator."""

    leaf_size: int = 16
    recent_window: int = 512
    summary_slots: int = 4
    address_context_left: int = 0
    address_context_right: int = 0
    revived_top_k: int = 4
    revived_min_k: int | None = None
    dynamic_score_margin: float = 0.0
    slot_score_reduction: str = "max"
    slot_score_temperature: float = 1.0
    symbolic_weight: float = 0.0
    symbolic_hash_bins: int = 512
    symbolic_ngram: int = 3
    symbolic_ngrams: tuple[int, ...] | None = None
    symbolic_ngram_weights: tuple[float, ...] | None = None
    symbolic_gate_top_k: int | None = None
    symbolic_context_left: int = 0
    symbolic_context_right: int = 0
    symbolic_score_norm: str = "none"
    revival_selection: str = "topk"
    diversity_weight: float = 0.0
    diversity_candidate_multiplier: int = 4
    gated_rerank_top_k: int | None = None
    channel_mix_neural_k: int = 1
    channel_mix_symbolic_k: int = 1
    rank_fusion_neural_weight: float = 1.0
    rank_fusion_symbolic_weight: float = 1.0
    rank_fusion_combined_weight: float = 1.0
    rank_fusion_offset: float = 8.0
    adaptive_budget_mode: str = "none"
    adaptive_min_k: int | None = None
    adaptive_entropy_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be positive")
        if self.recent_window < 0:
            raise ValueError("recent_window cannot be negative")
        if self.summary_slots <= 0:
            raise ValueError("summary_slots must be positive")
        if self.address_context_left < 0 or self.address_context_right < 0:
            raise ValueError("address context sizes cannot be negative")
        if self.revived_top_k <= 0:
            raise ValueError("revived_top_k must be positive")
        if self.revived_min_k is not None:
            if self.revived_min_k <= 0:
                raise ValueError("revived_min_k must be positive when provided")
            if self.revived_min_k > self.revived_top_k:
                raise ValueError("revived_min_k cannot exceed revived_top_k")
        if self.dynamic_score_margin < 0:
            raise ValueError("dynamic_score_margin cannot be negative")
        if self.slot_score_reduction not in {"max", "logsumexp", "mean"}:
            raise ValueError("slot_score_reduction must be 'max', 'logsumexp', or 'mean'")
        if self.slot_score_temperature <= 0:
            raise ValueError("slot_score_temperature must be positive")
        if self.symbolic_weight < 0:
            raise ValueError("symbolic_weight cannot be negative")
        if self.symbolic_hash_bins <= 0:
            raise ValueError("symbolic_hash_bins must be positive")
        if self.symbolic_ngram <= 0:
            raise ValueError("symbolic_ngram must be positive")
        if self.symbolic_ngrams is not None:
            if len(self.symbolic_ngrams) == 0:
                raise ValueError("symbolic_ngrams cannot be empty")
            if any(value <= 0 for value in self.symbolic_ngrams):
                raise ValueError("all symbolic_ngrams values must be positive")
        if self.symbolic_ngram_weights is not None:
            if self.symbolic_ngrams is None:
                raise ValueError("symbolic_ngram_weights requires symbolic_ngrams")
            if len(self.symbolic_ngram_weights) != len(self.symbolic_ngrams):
                raise ValueError("symbolic_ngram_weights must match symbolic_ngrams length")
            if any(value < 0 for value in self.symbolic_ngram_weights):
                raise ValueError("symbolic_ngram_weights cannot contain negative values")
            if sum(self.symbolic_ngram_weights) <= 0:
                raise ValueError("symbolic_ngram_weights must have positive sum")
        if self.symbolic_gate_top_k is not None and self.symbolic_gate_top_k <= 0:
            raise ValueError("symbolic_gate_top_k must be positive when provided")
        if self.symbolic_context_left < 0 or self.symbolic_context_right < 0:
            raise ValueError("symbolic context sizes cannot be negative")
        if self.symbolic_score_norm not in {"none", "zscore", "minmax"}:
            raise ValueError("symbolic_score_norm must be 'none', 'zscore', or 'minmax'")
        if self.revival_selection not in {
            "topk",
            "diverse",
            "channel-mix",
            "rank-fusion",
            "gated-rerank",
        }:
            raise ValueError(
                "revival_selection must be 'topk', 'diverse', 'channel-mix', "
                "'rank-fusion', or 'gated-rerank'"
            )
        if self.diversity_weight < 0:
            raise ValueError("diversity_weight cannot be negative")
        if self.diversity_candidate_multiplier <= 0:
            raise ValueError("diversity_candidate_multiplier must be positive")
        if self.gated_rerank_top_k is not None and self.gated_rerank_top_k <= 0:
            raise ValueError("gated_rerank_top_k must be positive when provided")
        if self.channel_mix_neural_k < 0:
            raise ValueError("channel_mix_neural_k cannot be negative")
        if self.channel_mix_symbolic_k < 0:
            raise ValueError("channel_mix_symbolic_k cannot be negative")
        if self.rank_fusion_neural_weight < 0:
            raise ValueError("rank_fusion_neural_weight cannot be negative")
        if self.rank_fusion_symbolic_weight < 0:
            raise ValueError("rank_fusion_symbolic_weight cannot be negative")
        if self.rank_fusion_combined_weight < 0:
            raise ValueError("rank_fusion_combined_weight cannot be negative")
        if (
            self.rank_fusion_neural_weight
            + self.rank_fusion_symbolic_weight
            + self.rank_fusion_combined_weight
            <= 0
        ):
            raise ValueError("at least one rank-fusion weight must be positive")
        if self.rank_fusion_offset <= 0:
            raise ValueError("rank_fusion_offset must be positive")
        if self.adaptive_budget_mode not in {"none", "entropy"}:
            raise ValueError("adaptive_budget_mode must be 'none' or 'entropy'")
        if self.adaptive_min_k is not None:
            if self.adaptive_min_k <= 0:
                raise ValueError("adaptive_min_k must be positive when provided")
            if self.adaptive_min_k > self.revived_top_k:
                raise ValueError("adaptive_min_k cannot exceed revived_top_k")
        if not 0.0 <= self.adaptive_entropy_threshold <= 1.0:
            raise ValueError("adaptive_entropy_threshold must be in [0, 1]")


@dataclass(frozen=True)
class RevivalCacheOutput:
    """Selected exact-token candidates plus cache diagnostics."""

    candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    revived_leaf_indices: torch.Tensor
    revived_leaf_mask: torch.Tensor
    leaf_scores: torch.Tensor
    leaf_summaries: torch.Tensor
    leaf_token_indices: torch.Tensor
    leaf_token_mask: torch.Tensor
    recent_start: int
    neural_leaf_scores: torch.Tensor | None = None
    symbolic_leaf_scores: torch.Tensor | None = None

    @property
    def active_exact_tokens_per_query(self) -> torch.Tensor:
        return self.candidate_mask.sum(dim=-1)


@dataclass(frozen=True)
class MemoryCacheProbeMetrics:
    """Metrics for source-hit survival experiments."""

    source_hit_rate: float
    source_leaf_accuracy: float
    source_leaf_loss: float
    source_survival_loss: float
    mean_active_exact_tokens: float
    max_active_exact_tokens: int
    old_leaf_count: int
    active_token_compression_ratio: float
    source_coverage_loss: float | None = None
    source_set_hit_rate: float | None = None
    source_set_leaf_accuracy: float | None = None
    revived_diversity_loss: float | None = None
    combined_source_rank: float | None = None
    neural_source_rank: float | None = None
    symbolic_source_rank: float | None = None
    neural_source_hit_rate: float | None = None
    symbolic_source_hit_rate: float | None = None


class MultiSlotLeafCompressor(nn.Module):
    """Compress each old leaf into several learned summary slots."""

    def __init__(self, hidden_size: int, summary_slots: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("MultiSlotLeafCompressor requires torch")
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if summary_slots <= 0:
            raise ValueError("summary_slots must be positive")
        self.hidden_size = hidden_size
        self.summary_slots = summary_slots
        self.slot_queries = nn.Parameter(torch.empty(summary_slots, hidden_size))
        nn.init.normal_(self.slot_queries, mean=0.0, std=hidden_size**-0.5)

    def forward(
        self,
        leaf_states: torch.Tensor,
        leaf_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return summaries with shape [batch, leaves, slots, hidden]."""

        if leaf_states.ndim != 4:
            raise ValueError("leaf_states must have shape [batch, leaves, leaf_size, hidden]")
        if leaf_mask.shape != leaf_states.shape[:3]:
            raise ValueError("leaf_mask must have shape [batch, leaves, leaf_size]")
        scores = torch.einsum("blth,sh->blts", leaf_states, self.slot_queries)
        scores = scores / math.sqrt(float(self.hidden_size))
        scores = scores.masked_fill(~leaf_mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=2)
        weights = torch.where(leaf_mask.unsqueeze(-1), weights, torch.zeros_like(weights))
        normalizer = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        weights = weights / normalizer
        return torch.einsum("blts,blth->blsh", weights, leaf_states)


class AverageLeafCompressor(nn.Module):
    """One-slot average baseline for old-leaf summaries."""

    def __init__(self, hidden_size: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("AverageLeafCompressor requires torch")
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.summary_slots = 1

    def forward(self, leaf_states: torch.Tensor, leaf_mask: torch.Tensor) -> torch.Tensor:
        if leaf_states.ndim != 4:
            raise ValueError("leaf_states must have shape [batch, leaves, leaf_size, hidden]")
        if leaf_mask.shape != leaf_states.shape[:3]:
            raise ValueError("leaf_mask must have shape [batch, leaves, leaf_size]")
        weights = leaf_mask.to(dtype=leaf_states.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        summary = torch.einsum("blt,blth->blh", weights / normalizer, leaf_states)
        return summary.unsqueeze(2)


class HybridLeafCompressor(nn.Module):
    """Average anchor plus learned residual slots for each old leaf."""

    def __init__(self, hidden_size: int, summary_slots: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("HybridLeafCompressor requires torch")
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if summary_slots <= 0:
            raise ValueError("summary_slots must be positive")
        self.hidden_size = hidden_size
        self.summary_slots = summary_slots
        self.average = AverageLeafCompressor(hidden_size)
        self.learned = (
            MultiSlotLeafCompressor(hidden_size, summary_slots - 1)
            if summary_slots > 1
            else None
        )

    def forward(self, leaf_states: torch.Tensor, leaf_mask: torch.Tensor) -> torch.Tensor:
        average_summary = self.average(leaf_states, leaf_mask)
        if self.learned is None:
            return average_summary
        learned_summaries = self.learned(leaf_states, leaf_mask)
        return torch.cat([average_summary, learned_summaries], dim=2)


class RevivalMemoryCache(nn.Module):
    """Route queries through summaries and return exact recent + revived tokens."""

    def __init__(
        self,
        hidden_size: int,
        config: RevivalCacheConfig,
        *,
        compressor: nn.Module | None = None,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("RevivalMemoryCache requires torch")
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.compressor = compressor or MultiSlotLeafCompressor(hidden_size, config.summary_slots)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        *,
        token_ids: torch.Tensor | None = None,
        query_token_ids: torch.Tensor | None = None,
    ) -> RevivalCacheOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, seq_len, hidden]")
        if query_states.ndim != 3:
            raise ValueError("query_states must have shape [batch, queries, hidden]")
        if hidden_states.shape[0] != query_states.shape[0]:
            raise ValueError("hidden_states and query_states batch sizes must match")
        if hidden_states.shape[-1] != self.hidden_size or query_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden size mismatch")
        if token_ids is not None and token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("token_ids must have shape [batch, seq_len]")
        if query_token_ids is not None and query_token_ids.shape[:2] != query_states.shape[:2]:
            raise ValueError("query_token_ids must have shape [batch, queries, window]")

        batch_size, seq_len, _ = hidden_states.shape
        recent_start = max(0, seq_len - self.config.recent_window)
        old_states = hidden_states[:, :recent_start]
        leaf_states, leaf_token_indices, leaf_mask = _split_old_tokens_into_leaves(
            old_states,
            leaf_size=self.config.leaf_size,
        )
        address_states, address_mask = (
            _split_old_tokens_into_context_windows(
                old_states,
                leaf_size=self.config.leaf_size,
                context_left=self.config.address_context_left,
                context_right=self.config.address_context_right,
            )
            if self.config.address_context_left or self.config.address_context_right
            else (leaf_states, leaf_mask)
        )

        if leaf_states.shape[1] == 0:
            leaf_summaries = hidden_states.new_zeros(
                batch_size,
                0,
                self.config.summary_slots,
                self.hidden_size,
            )
            leaf_scores = hidden_states.new_zeros(batch_size, query_states.shape[1], 0)
            neural_leaf_scores = leaf_scores
            symbolic_leaf_scores = None
            revived_leaf_indices = torch.zeros(
                batch_size,
                query_states.shape[1],
                0,
                dtype=torch.long,
                device=hidden_states.device,
            )
            revived_leaf_mask = torch.zeros_like(revived_leaf_indices, dtype=torch.bool)
        else:
            leaf_summaries = self.compressor(address_states, address_mask)
            slot_scores = torch.einsum("bqh,blsh->bqls", query_states, leaf_summaries)
            slot_scores = slot_scores / math.sqrt(float(self.hidden_size))
            leaf_scores = _reduce_slot_scores(
                slot_scores,
                reduction=self.config.slot_score_reduction,
                temperature=self.config.slot_score_temperature,
            )
            neural_leaf_scores = leaf_scores
            symbolic_leaf_scores = None
            if self.config.symbolic_weight and token_ids is not None and query_token_ids is not None:
                leaf_token_ids, token_leaf_mask = (
                    _split_old_token_ids_into_context_windows(
                        token_ids[:, :recent_start],
                        leaf_size=self.config.leaf_size,
                        context_left=self.config.symbolic_context_left,
                        context_right=self.config.symbolic_context_right,
                    )
                    if self.config.symbolic_context_left or self.config.symbolic_context_right
                    else _split_old_token_ids_into_leaves(
                        token_ids[:, :recent_start],
                        leaf_size=self.config.leaf_size,
                    )
                )
                symbolic_scores = _symbolic_leaf_scores(
                    query_token_ids,
                    leaf_token_ids,
                    token_leaf_mask,
                    hash_bins=self.config.symbolic_hash_bins,
                    ngrams=self.config.symbolic_ngrams or (self.config.symbolic_ngram,),
                    weights=self.config.symbolic_ngram_weights,
                    dtype=leaf_scores.dtype,
                )
                symbolic_scores = _normalize_scores(
                    symbolic_scores,
                    mode=self.config.symbolic_score_norm,
                )
                symbolic_leaf_scores = symbolic_scores
                if self.config.symbolic_gate_top_k is not None:
                    symbolic_scores = _gate_scores_by_topk(
                        symbolic_scores,
                        neural_leaf_scores,
                        top_k=self.config.symbolic_gate_top_k,
                    )
                    symbolic_leaf_scores = symbolic_scores
                leaf_scores = leaf_scores + self.config.symbolic_weight * symbolic_scores
            top_values, revived_leaf_indices = _select_revived_leaves(
                leaf_scores,
                leaf_summaries,
                neural_leaf_scores=neural_leaf_scores,
                symbolic_leaf_scores=symbolic_leaf_scores,
                top_k=min(self.config.revived_top_k, leaf_scores.shape[-1]),
                selection=self.config.revival_selection,
                diversity_weight=self.config.diversity_weight,
                diversity_candidate_multiplier=self.config.diversity_candidate_multiplier,
                gated_rerank_top_k=self.config.gated_rerank_top_k,
                channel_mix_neural_k=self.config.channel_mix_neural_k,
                channel_mix_symbolic_k=self.config.channel_mix_symbolic_k,
                rank_fusion_neural_weight=self.config.rank_fusion_neural_weight,
                rank_fusion_symbolic_weight=self.config.rank_fusion_symbolic_weight,
                rank_fusion_combined_weight=self.config.rank_fusion_combined_weight,
                rank_fusion_offset=self.config.rank_fusion_offset,
            )
            revived_leaf_mask = _revived_leaf_mask_from_top_values(
                top_values,
                min_k=self.config.revived_min_k,
                margin=self.config.dynamic_score_margin,
            )
            revived_leaf_mask = _apply_adaptive_budget_mask(
                revived_leaf_mask,
                leaf_scores,
                top_values,
                adaptive_mode=self.config.adaptive_budget_mode,
                adaptive_min_k=self.config.adaptive_min_k,
                entropy_threshold=self.config.adaptive_entropy_threshold,
            )

        candidate_indices, candidate_mask = _candidate_indices_from_recent_and_leaves(
            seq_len=seq_len,
            recent_start=recent_start,
            leaf_token_indices=leaf_token_indices,
            leaf_token_mask=leaf_mask,
            revived_leaf_indices=revived_leaf_indices,
            revived_leaf_mask=revived_leaf_mask,
            batch_size=batch_size,
            query_count=query_states.shape[1],
            device=hidden_states.device,
        )
        return RevivalCacheOutput(
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            revived_leaf_indices=revived_leaf_indices,
            revived_leaf_mask=revived_leaf_mask,
            leaf_scores=leaf_scores,
            leaf_summaries=leaf_summaries,
            leaf_token_indices=leaf_token_indices,
            leaf_token_mask=leaf_mask,
            recent_start=recent_start,
            neural_leaf_scores=neural_leaf_scores,
            symbolic_leaf_scores=symbolic_leaf_scores,
        )


class MemoryCacheRouteProbe(nn.Module):
    """Small trainable source-hit probe for the revival cache.

    It is deliberately not a language model. The probe learns byte embeddings,
    leaf summaries, and query pooling so we can test whether the cache can
    preserve answer-source leaves under a fixed active-token budget.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int,
        cache_config: RevivalCacheConfig,
        query_window: int = 96,
        compressor_type: str = "multislot",
        query_pooling: str = "mean",
        use_query_index_embedding: bool = False,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("MemoryCacheRouteProbe requires torch")
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if query_window <= 0:
            raise ValueError("query_window must be positive")
        if query_pooling not in {"mean", "last", "attn", "mean-pos"}:
            raise ValueError("query_pooling must be 'mean', 'last', 'attn', or 'mean-pos'")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.query_window = query_window
        self.query_pooling = query_pooling
        self.use_query_index_embedding = use_query_index_embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.state_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.query_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.query_position_embedding = (
            nn.Embedding(query_window, hidden_size)
            if query_pooling in {"attn", "mean-pos"}
            else None
        )
        self.query_pool_score = (
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )
            if query_pooling == "attn"
            else None
        )
        self.query_index_embedding = (
            nn.Embedding(max_seq_len, hidden_size)
            if use_query_index_embedding
            else None
        )
        if compressor_type == "multislot":
            compressor = MultiSlotLeafCompressor(hidden_size, cache_config.summary_slots)
        elif compressor_type == "average":
            compressor = AverageLeafCompressor(hidden_size)
        elif compressor_type == "hybrid":
            compressor = HybridLeafCompressor(hidden_size, cache_config.summary_slots)
        else:
            raise ValueError("compressor_type must be 'multislot', 'average', or 'hybrid'")
        self.cache = RevivalMemoryCache(
            hidden_size=hidden_size,
            config=cache_config,
            compressor=compressor,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        answer_indices: torch.Tensor,
    ) -> RevivalCacheOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")
        answer_indices = _ensure_query_dim(answer_indices).to(input_ids.device)
        token_states = self.token_embedding(input_ids)
        hidden_states = self.state_projection(token_states)
        query_context = self._pool_query_context(hidden_states, answer_indices)
        if self.query_index_embedding is not None:
            query_ordinals = torch.arange(
                answer_indices.shape[1],
                device=input_ids.device,
            ).clamp(max=self.max_seq_len - 1)
            query_context = query_context + self.query_index_embedding(query_ordinals).unsqueeze(0)
        query_token_ids = _causal_window_token_ids(
            input_ids,
            answer_indices,
            window_size=self.query_window,
        )
        query_states = self.query_projection(query_context)
        return self.cache(
            hidden_states,
            query_states,
            token_ids=input_ids,
            query_token_ids=query_token_ids,
        )

    def _pool_query_context(
        self,
        hidden_states: torch.Tensor,
        answer_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.query_pooling == "mean":
            return _causal_window_mean(
                hidden_states,
                answer_indices,
                window_size=self.query_window,
            )
        if self.query_pooling == "last":
            return _gather_positions(hidden_states, answer_indices)
        window_states, window_mask = _causal_window_states(
            hidden_states,
            answer_indices,
            window_size=self.query_window,
        )
        if self.query_position_embedding is not None:
            positions = torch.arange(self.query_window, device=hidden_states.device)
            window_states = window_states + self.query_position_embedding(positions).view(
                1,
                1,
                self.query_window,
                self.hidden_size,
            )
        if self.query_pooling == "mean-pos":
            weights = window_mask.to(dtype=hidden_states.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return torch.einsum("bqw,bqwh->bqh", weights, window_states)
        if self.query_pooling != "attn" or self.query_pool_score is None:
            raise ValueError(f"unknown query pooling mode: {self.query_pooling}")
        scores = self.query_pool_score(window_states).squeeze(-1)
        scores = scores.masked_fill(~window_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(window_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bqw,bqwh->bqh", weights, window_states)


class UsefulnessLedger:
    """Small CPU-side usefulness tracker for revived leaves."""

    def __init__(self, leaf_count: int, *, decay: float = 0.95, revive_bonus: float = 1.0) -> None:
        if torch is None:
            raise ModuleNotFoundError("UsefulnessLedger requires torch")
        if leaf_count < 0:
            raise ValueError("leaf_count cannot be negative")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if revive_bonus < 0:
            raise ValueError("revive_bonus cannot be negative")
        self.scores = torch.zeros(leaf_count, dtype=torch.float32)
        self.decay = decay
        self.revive_bonus = revive_bonus

    def update(self, revived_leaf_indices: torch.Tensor, revived_leaf_mask: torch.Tensor) -> None:
        if revived_leaf_indices.shape != revived_leaf_mask.shape:
            raise ValueError("revived leaf indices and mask must have matching shapes")
        self.scores.mul_(self.decay)
        selected = revived_leaf_indices.detach().cpu()[revived_leaf_mask.detach().cpu()]
        if selected.numel() == 0:
            return
        selected = selected[(selected >= 0) & (selected < self.scores.numel())]
        if selected.numel() == 0:
            return
        self.scores.index_add_(
            0,
            selected,
            torch.full((selected.numel(),), self.revive_bonus, dtype=self.scores.dtype),
        )

    def lowest_scoring_leaves(self, count: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        if self.scores.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        return torch.topk(self.scores, k=min(count, self.scores.numel()), largest=False).indices


def scheduled_recent_window(
    *,
    start_window: int,
    end_window: int,
    step: int,
    curriculum_steps: int,
) -> int:
    """Linear cache-diet schedule for exact recent-window size."""

    if start_window < 0 or end_window < 0:
        raise ValueError("window sizes cannot be negative")
    if step <= 0:
        raise ValueError("step must be positive")
    if curriculum_steps <= 0:
        return end_window
    progress = min(max((step - 1) / float(curriculum_steps - 1), 0.0), 1.0)
    value = start_window + progress * (end_window - start_window)
    return int(round(value))


def memory_cache_source_leaf_loss(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor:
    """Cross-entropy routing loss toward the leaf containing each source token."""

    if torch is None or F is None:
        raise ModuleNotFoundError("memory_cache_source_leaf_loss requires torch")
    if output.leaf_scores.ndim != 3:
        raise ValueError("leaf_scores must have shape [batch, queries, leaves]")
    if output.leaf_scores.shape[-1] == 0:
        return output.leaf_scores.sum() * 0.0
    targets, valid = _source_leaf_targets(output, source_indices, leaf_size=leaf_size)
    if not valid.any():
        return output.leaf_scores.sum() * 0.0
    safe_targets = targets.clamp(min=0, max=output.leaf_scores.shape[-1] - 1)
    losses = F.cross_entropy(
        output.leaf_scores.reshape(-1, output.leaf_scores.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).view_as(safe_targets)
    return losses[valid].mean()


def memory_cache_topk_survival_loss(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
    top_k: int,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable loss for keeping the source leaf inside revived top-k.

    Cross-entropy optimizes rank 1. Revival only needs the source leaf to be
    inside the revived set, so this loss compares the source score against the
    kth strongest non-source leaf.
    """

    if torch is None or F is None:
        raise ModuleNotFoundError("memory_cache_topk_survival_loss requires torch")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if output.leaf_scores.ndim != 3:
        raise ValueError("leaf_scores must have shape [batch, queries, leaves]")
    leaf_count = output.leaf_scores.shape[-1]
    if leaf_count == 0 or top_k >= leaf_count:
        return output.leaf_scores.sum() * 0.0
    targets, valid = _source_leaf_targets(output, source_indices, leaf_size=leaf_size)
    if not valid.any():
        return output.leaf_scores.sum() * 0.0

    safe_targets = targets.clamp(min=0, max=leaf_count - 1)
    source_scores = output.leaf_scores.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    leaf_ids = torch.arange(leaf_count, device=output.leaf_scores.device).view(1, 1, leaf_count)
    negative_scores = output.leaf_scores.masked_fill(
        leaf_ids == safe_targets.unsqueeze(-1),
        torch.finfo(output.leaf_scores.dtype).min,
    )
    kth_negative = torch.topk(negative_scores, k=top_k, dim=-1).values[..., -1]
    loss = F.softplus((kth_negative + margin - source_scores) / temperature) * temperature
    return loss[valid].mean()


def memory_cache_source_set_leaf_loss(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor:
    """Multi-positive routing loss for tasks with several relevant source leaves."""

    if torch is None or F is None:
        raise ModuleNotFoundError("memory_cache_source_set_leaf_loss requires torch")
    if output.leaf_scores.ndim != 3:
        raise ValueError("leaf_scores must have shape [batch, queries, leaves]")
    if output.leaf_scores.shape[-1] == 0:
        return output.leaf_scores.sum() * 0.0
    target_mask = _source_leaf_target_mask_from_scores(
        output.leaf_scores,
        source_indices,
        leaf_size=leaf_size,
    )
    valid = target_mask.any(dim=-1)
    if not valid.any():
        return output.leaf_scores.sum() * 0.0
    min_value = torch.finfo(output.leaf_scores.dtype).min
    log_all = torch.logsumexp(output.leaf_scores, dim=-1)
    log_positive = torch.logsumexp(output.leaf_scores.masked_fill(~target_mask, min_value), dim=-1)
    return (log_all - log_positive)[valid].mean()


def memory_cache_source_set_survival_loss(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
    top_k: int,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Push every source leaf in a target set above the strongest negative leaves."""

    if torch is None or F is None:
        raise ModuleNotFoundError("memory_cache_source_set_survival_loss requires torch")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if output.leaf_scores.ndim != 3:
        raise ValueError("leaf_scores must have shape [batch, queries, leaves]")
    leaf_count = output.leaf_scores.shape[-1]
    if leaf_count == 0 or top_k >= leaf_count:
        return output.leaf_scores.sum() * 0.0
    target_mask = _source_leaf_target_mask_from_scores(
        output.leaf_scores,
        source_indices,
        leaf_size=leaf_size,
    )
    valid = target_mask.any(dim=-1)
    has_negative = (~target_mask).any(dim=-1)
    loss_mask = target_mask & has_negative.unsqueeze(-1)
    if not valid.any() or not loss_mask.any():
        return output.leaf_scores.sum() * 0.0

    min_value = torch.finfo(output.leaf_scores.dtype).min
    negative_scores = output.leaf_scores.masked_fill(target_mask, min_value)
    kth_negative = torch.topk(negative_scores, k=min(top_k, leaf_count), dim=-1).values[..., -1]
    loss = (
        F.softplus((kth_negative.unsqueeze(-1) + margin - output.leaf_scores) / temperature)
        * temperature
    )
    return loss[loss_mask].mean()


def memory_cache_source_set_coverage_loss(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Uniform-target CE over every source leaf in a multi-source answer set.

    Multi-positive logsumexp loss rewards any relevant leaf. Aggregation needs
    coverage, so this loss gives each unique source leaf equal probability mass.
    """

    if torch is None or F is None:
        raise ModuleNotFoundError("memory_cache_source_set_coverage_loss requires torch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if output.leaf_scores.ndim != 3:
        raise ValueError("leaf_scores must have shape [batch, queries, leaves]")
    if output.leaf_scores.shape[-1] == 0:
        return output.leaf_scores.sum() * 0.0
    target_mask = _source_leaf_target_mask_from_scores(
        output.leaf_scores,
        source_indices,
        leaf_size=leaf_size,
    )
    valid = target_mask.any(dim=-1)
    if not valid.any():
        return output.leaf_scores.sum() * 0.0
    target = target_mask.to(dtype=output.leaf_scores.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(output.leaf_scores / temperature, dim=-1)
    loss = -(target * log_probs).sum(dim=-1) * temperature
    return loss[valid].mean()


def memory_cache_summary_diversity_loss(output: RevivalCacheOutput) -> torch.Tensor:
    """Penalize collapsed summary slots inside each old leaf."""

    if torch is None or F is None:
        raise ModuleNotFoundError("memory_cache_summary_diversity_loss requires torch")
    summaries = output.leaf_summaries
    if summaries.ndim != 4:
        raise ValueError("leaf_summaries must have shape [batch, leaves, slots, hidden]")
    if summaries.shape[1] == 0 or summaries.shape[2] <= 1:
        return summaries.sum() * 0.0
    normalized = F.normalize(summaries, dim=-1)
    gram = torch.einsum("blsh,blth->blst", normalized, normalized)
    slots = summaries.shape[2]
    eye = torch.eye(slots, device=summaries.device, dtype=torch.bool).view(1, 1, slots, slots)
    off_diagonal = gram.masked_select((~eye).expand_as(gram))
    return off_diagonal.pow(2).mean()


def memory_cache_revived_leaf_diversity_loss(output: RevivalCacheOutput) -> torch.Tensor:
    """Penalize highly similar summaries among revived leaves for each query."""

    if torch is None or F is None:
        raise ModuleNotFoundError("memory_cache_revived_leaf_diversity_loss requires torch")
    summaries = output.leaf_summaries
    if summaries.ndim != 4:
        raise ValueError("leaf_summaries must have shape [batch, leaves, slots, hidden]")
    if summaries.shape[1] == 0 or output.revived_leaf_indices.shape[-1] <= 1:
        return summaries.sum() * 0.0
    leaf_vectors = F.normalize(summaries.mean(dim=2), dim=-1)
    query_count = output.revived_leaf_indices.shape[1]
    expanded_vectors = leaf_vectors.unsqueeze(1).expand(
        leaf_vectors.shape[0],
        query_count,
        leaf_vectors.shape[1],
        leaf_vectors.shape[-1],
    )
    safe_indices = output.revived_leaf_indices.clamp(min=0, max=leaf_vectors.shape[1] - 1)
    selected_vectors = expanded_vectors.gather(
        2,
        safe_indices.unsqueeze(-1).expand(*safe_indices.shape, leaf_vectors.shape[-1]),
    )
    selected_mask = output.revived_leaf_mask
    pair_mask = selected_mask.unsqueeze(-1) & selected_mask.unsqueeze(-2)
    slots = safe_indices.shape[-1]
    eye = torch.eye(slots, dtype=torch.bool, device=safe_indices.device).view(1, 1, slots, slots)
    pair_mask = pair_mask & ~eye
    if not pair_mask.any():
        return summaries.sum() * 0.0
    cosine = torch.einsum("bqkh,bqmh->bqkm", selected_vectors, selected_vectors)
    return cosine.clamp_min(0.0).pow(2)[pair_mask].mean()


def memory_cache_source_leaf_accuracy(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor:
    source_indices = _ensure_query_dim(source_indices).to(output.leaf_scores.device)
    if output.leaf_scores.shape[-1] == 0:
        return output.leaf_scores.new_tensor(0.0)
    targets = torch.div(source_indices, leaf_size, rounding_mode="floor")
    valid = (targets >= 0) & (targets < output.leaf_scores.shape[-1])
    if not valid.any():
        return output.leaf_scores.new_tensor(0.0)
    predictions = output.leaf_scores.argmax(dim=-1)
    return (predictions[valid] == targets[valid]).float().mean()


def memory_cache_source_set_leaf_accuracy(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor:
    """Top-1 leaf accuracy where any leaf in the source set is accepted."""

    if output.leaf_scores.shape[-1] == 0:
        return output.leaf_scores.new_tensor(0.0)
    target_mask = _source_leaf_target_mask_from_scores(
        output.leaf_scores,
        source_indices,
        leaf_size=leaf_size,
    )
    valid = target_mask.any(dim=-1)
    if not valid.any():
        return output.leaf_scores.new_tensor(0.0)
    predictions = output.leaf_scores.argmax(dim=-1)
    hit = target_mask.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    return hit[valid].float().mean()


def memory_cache_source_hit_rate(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
) -> torch.Tensor:
    """Fraction of source tokens present in exact recent + revived candidates."""

    source_indices = _ensure_query_dim(source_indices).to(output.candidate_indices.device)
    if source_indices.shape != output.candidate_indices.shape[:2]:
        raise ValueError("source_indices must have shape [batch, queries]")
    hit = (output.candidate_indices == source_indices.unsqueeze(-1)) & output.candidate_mask
    return hit.any(dim=-1).float().mean()


def memory_cache_source_set_leaf_recall(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor:
    """Mean fraction of relevant old source leaves present in exact candidates."""

    target_mask = _source_leaf_target_mask_from_scores(
        output.leaf_scores,
        source_indices,
        leaf_size=leaf_size,
    )
    valid = target_mask.any(dim=-1)
    if not valid.any():
        return output.leaf_scores.new_tensor(0.0)
    candidate_mask = _candidate_leaf_mask(output, leaf_size=leaf_size)
    recalled = (target_mask & candidate_mask).sum(dim=-1).to(dtype=output.leaf_scores.dtype)
    total = target_mask.sum(dim=-1).clamp_min(1).to(dtype=output.leaf_scores.dtype)
    return (recalled / total)[valid].mean()


def memory_cache_score_source_hit_rate(
    scores: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
    top_k: int,
) -> torch.Tensor | None:
    """Fraction of source leaves inside a score tensor's top-k leaves."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, queries, leaves]")
    leaf_count = scores.shape[-1]
    if leaf_count == 0:
        return None
    targets, valid = _source_leaf_targets_from_scores(scores, source_indices, leaf_size=leaf_size)
    if not valid.any():
        return None
    safe_targets = targets.clamp(min=0, max=leaf_count - 1)
    top_k = min(max(top_k, 1), leaf_count)
    top_indices = torch.topk(scores, k=top_k, dim=-1).indices
    hit = top_indices == safe_targets.unsqueeze(-1)
    return hit.any(dim=-1)[valid].float().mean()


def memory_cache_score_source_set_hit_rate(
    scores: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
    top_k: int,
) -> torch.Tensor | None:
    """Top-k recall for a set of relevant source leaves."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, queries, leaves]")
    leaf_count = scores.shape[-1]
    if leaf_count == 0:
        return None
    target_mask = _source_leaf_target_mask_from_scores(scores, source_indices, leaf_size=leaf_size)
    valid = target_mask.any(dim=-1)
    if not valid.any():
        return None
    top_k = min(max(top_k, 1), leaf_count)
    top_indices = torch.topk(scores, k=top_k, dim=-1).indices
    top_mask = torch.zeros_like(target_mask)
    top_mask.scatter_(-1, top_indices, True)
    recalled = (target_mask & top_mask).sum(dim=-1).to(dtype=scores.dtype)
    total = target_mask.sum(dim=-1).clamp_min(1).to(dtype=scores.dtype)
    return (recalled / total)[valid].mean()


def memory_cache_score_source_rank(
    scores: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor | None:
    """Mean 1-indexed rank of source leaves under a score tensor."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, queries, leaves]")
    leaf_count = scores.shape[-1]
    if leaf_count == 0:
        return None
    targets, valid = _source_leaf_targets_from_scores(scores, source_indices, leaf_size=leaf_size)
    if not valid.any():
        return None
    safe_targets = targets.clamp(min=0, max=leaf_count - 1)
    source_scores = scores.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    rank = (scores > source_scores.unsqueeze(-1)).sum(dim=-1) + 1
    return rank[valid].float().mean()


def memory_cache_score_source_set_rank(
    scores: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor | None:
    """Mean 1-indexed rank over every leaf in the source set."""

    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, queries, leaves]")
    leaf_count = scores.shape[-1]
    if leaf_count == 0:
        return None
    target_mask = _source_leaf_target_mask_from_scores(scores, source_indices, leaf_size=leaf_size)
    if not target_mask.any():
        return None
    sorted_indices = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(sorted_indices)
    rank_values = torch.arange(1, leaf_count + 1, device=scores.device).view(1, 1, leaf_count)
    ranks.scatter_(-1, sorted_indices, rank_values.expand_as(sorted_indices))
    return ranks[target_mask].float().mean()


def collect_probe_metrics(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    seq_len: int,
    leaf_size: int,
) -> MemoryCacheProbeMetrics:
    leaf_loss = memory_cache_source_leaf_loss(
        output,
        source_indices,
        leaf_size=leaf_size,
    )
    leaf_accuracy = memory_cache_source_leaf_accuracy(
        output,
        source_indices,
        leaf_size=leaf_size,
    )
    survival_loss = memory_cache_topk_survival_loss(
        output,
        source_indices,
        leaf_size=leaf_size,
        top_k=max(int(output.revived_leaf_indices.shape[-1]), 1),
    )
    source_hit = memory_cache_source_hit_rate(output, source_indices)
    revived_diversity_loss = memory_cache_revived_leaf_diversity_loss(output)
    active = output.active_exact_tokens_per_query.float()
    mean_active = float(active.mean().item())
    diagnostic_top_k = max(int(output.revived_leaf_indices.shape[-1]), 1)
    combined_rank = memory_cache_score_source_rank(
        output.leaf_scores,
        source_indices,
        leaf_size=leaf_size,
    )
    neural_rank = (
        memory_cache_score_source_rank(
            output.neural_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
        )
        if output.neural_leaf_scores is not None
        else None
    )
    symbolic_rank = (
        memory_cache_score_source_rank(
            output.symbolic_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
        )
        if output.symbolic_leaf_scores is not None
        else None
    )
    neural_hit = (
        memory_cache_score_source_hit_rate(
            output.neural_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
            top_k=diagnostic_top_k,
        )
        if output.neural_leaf_scores is not None
        else None
    )
    symbolic_hit = (
        memory_cache_score_source_hit_rate(
            output.symbolic_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
            top_k=diagnostic_top_k,
        )
        if output.symbolic_leaf_scores is not None
        else None
    )
    return MemoryCacheProbeMetrics(
        source_hit_rate=float(source_hit.item()),
        source_leaf_accuracy=float(leaf_accuracy.item()),
        source_leaf_loss=float(leaf_loss.item()),
        source_survival_loss=float(survival_loss.item()),
        mean_active_exact_tokens=mean_active,
        max_active_exact_tokens=int(active.max().item()),
        old_leaf_count=int(output.leaf_summaries.shape[1]),
        active_token_compression_ratio=float(seq_len / max(mean_active, 1.0)),
        revived_diversity_loss=float(revived_diversity_loss.item()),
        combined_source_rank=_optional_tensor_to_float(combined_rank),
        neural_source_rank=_optional_tensor_to_float(neural_rank),
        symbolic_source_rank=_optional_tensor_to_float(symbolic_rank),
        neural_source_hit_rate=_optional_tensor_to_float(neural_hit),
        symbolic_source_hit_rate=_optional_tensor_to_float(symbolic_hit),
    )


def collect_probe_set_metrics(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    seq_len: int,
    leaf_size: int,
) -> MemoryCacheProbeMetrics:
    """Collect metrics for tasks where each query may need several source leaves."""

    leaf_loss = memory_cache_source_set_leaf_loss(
        output,
        source_indices,
        leaf_size=leaf_size,
    )
    coverage_loss = memory_cache_source_set_coverage_loss(
        output,
        source_indices,
        leaf_size=leaf_size,
    )
    leaf_accuracy = memory_cache_source_set_leaf_accuracy(
        output,
        source_indices,
        leaf_size=leaf_size,
    )
    survival_loss = memory_cache_source_set_survival_loss(
        output,
        source_indices,
        leaf_size=leaf_size,
        top_k=max(int(output.revived_leaf_indices.shape[-1]), 1),
    )
    source_hit = memory_cache_source_set_leaf_recall(
        output,
        source_indices,
        leaf_size=leaf_size,
    )
    revived_diversity_loss = memory_cache_revived_leaf_diversity_loss(output)
    active = output.active_exact_tokens_per_query.float()
    mean_active = float(active.mean().item())
    diagnostic_top_k = max(int(output.revived_leaf_indices.shape[-1]), 1)
    combined_rank = memory_cache_score_source_set_rank(
        output.leaf_scores,
        source_indices,
        leaf_size=leaf_size,
    )
    neural_rank = (
        memory_cache_score_source_set_rank(
            output.neural_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
        )
        if output.neural_leaf_scores is not None
        else None
    )
    symbolic_rank = (
        memory_cache_score_source_set_rank(
            output.symbolic_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
        )
        if output.symbolic_leaf_scores is not None
        else None
    )
    neural_hit = (
        memory_cache_score_source_set_hit_rate(
            output.neural_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
            top_k=diagnostic_top_k,
        )
        if output.neural_leaf_scores is not None
        else None
    )
    symbolic_hit = (
        memory_cache_score_source_set_hit_rate(
            output.symbolic_leaf_scores,
            source_indices,
            leaf_size=leaf_size,
            top_k=diagnostic_top_k,
        )
        if output.symbolic_leaf_scores is not None
        else None
    )
    return MemoryCacheProbeMetrics(
        source_hit_rate=float(source_hit.item()),
        source_leaf_accuracy=float(leaf_accuracy.item()),
        source_leaf_loss=float(leaf_loss.item()),
        source_survival_loss=float(survival_loss.item()),
        mean_active_exact_tokens=mean_active,
        max_active_exact_tokens=int(active.max().item()),
        old_leaf_count=int(output.leaf_summaries.shape[1]),
        active_token_compression_ratio=float(seq_len / max(mean_active, 1.0)),
        source_coverage_loss=float(coverage_loss.item()),
        source_set_hit_rate=float(source_hit.item()),
        source_set_leaf_accuracy=float(leaf_accuracy.item()),
        revived_diversity_loss=float(revived_diversity_loss.item()),
        combined_source_rank=_optional_tensor_to_float(combined_rank),
        neural_source_rank=_optional_tensor_to_float(neural_rank),
        symbolic_source_rank=_optional_tensor_to_float(symbolic_rank),
        neural_source_hit_rate=_optional_tensor_to_float(neural_hit),
        symbolic_source_hit_rate=_optional_tensor_to_float(symbolic_hit),
    )


def _split_old_tokens_into_leaves(
    old_states: torch.Tensor,
    *,
    leaf_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, old_len, hidden_size = old_states.shape
    if old_len == 0:
        empty_states = old_states.new_zeros(batch_size, 0, leaf_size, hidden_size)
        empty_indices = torch.zeros(
            0,
            leaf_size,
            dtype=torch.long,
            device=old_states.device,
        )
        empty_mask = torch.zeros(batch_size, 0, leaf_size, dtype=torch.bool, device=old_states.device)
        return empty_states, empty_indices, empty_mask
    leaf_count = math.ceil(old_len / leaf_size)
    padded_len = leaf_count * leaf_size
    pad_len = padded_len - old_len
    if pad_len:
        padding = old_states.new_zeros(batch_size, pad_len, hidden_size)
        old_states = torch.cat([old_states, padding], dim=1)
    leaf_states = old_states.view(batch_size, leaf_count, leaf_size, hidden_size)
    flat_indices = torch.arange(padded_len, device=old_states.device)
    leaf_token_indices = flat_indices.view(leaf_count, leaf_size)
    leaf_token_mask = leaf_token_indices < old_len
    leaf_token_mask = leaf_token_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return leaf_states, leaf_token_indices, leaf_token_mask


def _split_old_tokens_into_context_windows(
    old_states: torch.Tensor,
    *,
    leaf_size: int,
    context_left: int,
    context_right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, old_len, hidden_size = old_states.shape
    context_width = leaf_size + context_left + context_right
    if old_len == 0:
        empty_states = old_states.new_zeros(batch_size, 0, context_width, hidden_size)
        empty_mask = torch.zeros(batch_size, 0, context_width, dtype=torch.bool, device=old_states.device)
        return empty_states, empty_mask
    leaf_count = math.ceil(old_len / leaf_size)
    starts = torch.arange(leaf_count, device=old_states.device).unsqueeze(-1) * leaf_size
    offsets = torch.arange(
        -context_left,
        leaf_size + context_right,
        device=old_states.device,
    ).unsqueeze(0)
    indices = starts + offsets
    valid = (indices >= 0) & (indices < old_len)
    safe_indices = indices.clamp(min=0, max=old_len - 1)
    gathered = old_states[:, safe_indices.reshape(-1), :]
    gathered = gathered.view(batch_size, leaf_count, context_width, hidden_size)
    mask = valid.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.where(mask.unsqueeze(-1), gathered, torch.zeros_like(gathered)), mask


def _split_old_token_ids_into_leaves(
    old_token_ids: torch.Tensor,
    *,
    leaf_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, old_len = old_token_ids.shape
    if old_len == 0:
        empty_ids = torch.full(
            (batch_size, 0, leaf_size),
            -1,
            dtype=old_token_ids.dtype,
            device=old_token_ids.device,
        )
        empty_mask = torch.zeros(batch_size, 0, leaf_size, dtype=torch.bool, device=old_token_ids.device)
        return empty_ids, empty_mask
    leaf_count = math.ceil(old_len / leaf_size)
    padded_len = leaf_count * leaf_size
    pad_len = padded_len - old_len
    if pad_len:
        padding = torch.full(
            (batch_size, pad_len),
            -1,
            dtype=old_token_ids.dtype,
            device=old_token_ids.device,
        )
        old_token_ids = torch.cat([old_token_ids, padding], dim=1)
    leaf_ids = old_token_ids.view(batch_size, leaf_count, leaf_size)
    leaf_mask = leaf_ids >= 0
    return leaf_ids, leaf_mask


def _split_old_token_ids_into_context_windows(
    old_token_ids: torch.Tensor,
    *,
    leaf_size: int,
    context_left: int,
    context_right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, old_len = old_token_ids.shape
    context_width = leaf_size + context_left + context_right
    if old_len == 0:
        empty_ids = torch.full(
            (batch_size, 0, context_width),
            -1,
            dtype=old_token_ids.dtype,
            device=old_token_ids.device,
        )
        empty_mask = torch.zeros(batch_size, 0, context_width, dtype=torch.bool, device=old_token_ids.device)
        return empty_ids, empty_mask
    leaf_count = math.ceil(old_len / leaf_size)
    starts = torch.arange(leaf_count, device=old_token_ids.device).unsqueeze(-1) * leaf_size
    offsets = torch.arange(
        -context_left,
        leaf_size + context_right,
        device=old_token_ids.device,
    ).unsqueeze(0)
    indices = starts + offsets
    valid = (indices >= 0) & (indices < old_len)
    safe_indices = indices.clamp(min=0, max=old_len - 1)
    gathered = old_token_ids[:, safe_indices.reshape(-1)]
    gathered = gathered.view(batch_size, leaf_count, context_width)
    padding = torch.full_like(gathered, -1)
    mask = valid.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.where(mask, gathered, padding), mask


def _ensure_query_dim(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        return indices.unsqueeze(1)
    if indices.ndim == 2:
        return indices
    raise ValueError("indices must have shape [batch] or [batch, queries]")


def _ensure_source_set_dim(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        return indices.view(indices.shape[0], 1, 1)
    if indices.ndim == 2:
        return indices.unsqueeze(-1)
    if indices.ndim == 3:
        return indices
    raise ValueError("indices must have shape [batch], [batch, queries], or [batch, queries, sources]")


def _source_leaf_targets(
    output: RevivalCacheOutput,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_indices = _ensure_query_dim(source_indices).to(output.leaf_scores.device)
    if source_indices.shape != output.leaf_scores.shape[:2]:
        raise ValueError("source_indices must have shape [batch, queries]")
    targets = torch.div(source_indices, leaf_size, rounding_mode="floor")
    valid = (targets >= 0) & (targets < output.leaf_scores.shape[-1])
    return targets, valid


def _source_leaf_targets_from_scores(
    scores: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_indices = _ensure_query_dim(source_indices).to(scores.device)
    if source_indices.shape != scores.shape[:2]:
        raise ValueError("source_indices must have shape [batch, queries]")
    targets = torch.div(source_indices, leaf_size, rounding_mode="floor")
    valid = (targets >= 0) & (targets < scores.shape[-1])
    return targets, valid


def _source_leaf_target_mask_from_scores(
    scores: torch.Tensor,
    source_indices: torch.Tensor,
    *,
    leaf_size: int,
) -> torch.Tensor:
    source_indices = _ensure_source_set_dim(source_indices).to(scores.device)
    if source_indices.shape[:2] != scores.shape[:2]:
        raise ValueError("source_indices must have shape [batch, queries] or [batch, queries, sources]")
    leaf_count = scores.shape[-1]
    targets = torch.div(source_indices, leaf_size, rounding_mode="floor")
    valid = (targets >= 0) & (targets < leaf_count)
    safe_targets = targets.clamp(min=0, max=max(leaf_count - 1, 0))
    counts = torch.zeros(*scores.shape[:2], leaf_count, dtype=torch.long, device=scores.device)
    if leaf_count == 0:
        return counts.bool()
    counts.scatter_add_(-1, safe_targets, valid.to(dtype=counts.dtype))
    return counts > 0


def _candidate_leaf_mask(output: RevivalCacheOutput, *, leaf_size: int) -> torch.Tensor:
    leaf_count = output.leaf_scores.shape[-1]
    mask = torch.zeros(
        *output.leaf_scores.shape[:2],
        leaf_count,
        dtype=torch.long,
        device=output.leaf_scores.device,
    )
    if leaf_count == 0:
        return mask.bool()
    candidate_indices = output.candidate_indices.to(output.leaf_scores.device)
    candidate_valid = output.candidate_mask.to(output.leaf_scores.device)
    old_valid = candidate_valid & (candidate_indices >= 0) & (candidate_indices < output.recent_start)
    candidate_leaves = torch.div(candidate_indices, leaf_size, rounding_mode="floor")
    leaf_valid = old_valid & (candidate_leaves >= 0) & (candidate_leaves < leaf_count)
    safe_leaves = candidate_leaves.clamp(min=0, max=leaf_count - 1)
    mask.scatter_add_(-1, safe_leaves, leaf_valid.to(dtype=mask.dtype))
    return mask.bool()


def _optional_tensor_to_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.item())


def _reduce_slot_scores(
    slot_scores: torch.Tensor,
    *,
    reduction: str,
    temperature: float,
) -> torch.Tensor:
    if reduction == "max":
        return slot_scores.max(dim=-1).values
    if reduction == "mean":
        return slot_scores.mean(dim=-1)
    if reduction == "logsumexp":
        slots = slot_scores.shape[-1]
        smooth = temperature * torch.logsumexp(slot_scores / temperature, dim=-1)
        return smooth - temperature * math.log(float(max(slots, 1)))
    raise ValueError(f"unknown slot score reduction: {reduction}")


def _select_revived_leaves(
    leaf_scores: torch.Tensor,
    leaf_summaries: torch.Tensor,
    *,
    neural_leaf_scores: torch.Tensor | None,
    symbolic_leaf_scores: torch.Tensor | None,
    top_k: int,
    selection: str,
    diversity_weight: float,
    diversity_candidate_multiplier: int,
    gated_rerank_top_k: int | None,
    channel_mix_neural_k: int,
    channel_mix_symbolic_k: int,
    rank_fusion_neural_weight: float,
    rank_fusion_symbolic_weight: float,
    rank_fusion_combined_weight: float,
    rank_fusion_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        empty_shape = (*leaf_scores.shape[:2], 0)
        empty_indices = torch.zeros(empty_shape, dtype=torch.long, device=leaf_scores.device)
        return leaf_scores.new_zeros(empty_shape), empty_indices
    if selection == "topk" or (selection == "diverse" and diversity_weight == 0.0):
        return torch.topk(leaf_scores, k=top_k, dim=-1)
    if selection == "channel-mix":
        return _channel_mix_topk_leaves(
            leaf_scores,
            neural_leaf_scores,
            symbolic_leaf_scores,
            top_k=top_k,
            neural_k=channel_mix_neural_k,
            symbolic_k=channel_mix_symbolic_k,
            candidate_multiplier=diversity_candidate_multiplier,
        )
    if selection == "gated-rerank":
        return _gated_rerank_topk_leaves(
            leaf_scores,
            neural_leaf_scores,
            top_k=top_k,
            candidate_top_k=gated_rerank_top_k,
        )
    if selection == "rank-fusion":
        return _rank_fusion_topk_leaves(
            leaf_scores,
            neural_leaf_scores,
            symbolic_leaf_scores,
            top_k=top_k,
            neural_weight=rank_fusion_neural_weight,
            symbolic_weight=rank_fusion_symbolic_weight,
            combined_weight=rank_fusion_combined_weight,
            offset=rank_fusion_offset,
        )
    if selection != "diverse":
        raise ValueError(f"unknown revival selection: {selection}")
    return _diverse_topk_leaves(
        leaf_scores,
        leaf_summaries,
        top_k=top_k,
        diversity_weight=diversity_weight,
        candidate_multiplier=diversity_candidate_multiplier,
    )


def _channel_mix_topk_leaves(
    leaf_scores: torch.Tensor,
    neural_leaf_scores: torch.Tensor | None,
    symbolic_leaf_scores: torch.Tensor | None,
    *,
    top_k: int,
    neural_k: int,
    symbolic_k: int,
    candidate_multiplier: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    leaf_count = leaf_scores.shape[-1]
    candidate_parts = []
    if symbolic_leaf_scores is not None and symbolic_k > 0:
        k = min(symbolic_k, leaf_count)
        candidate_parts.append(torch.topk(symbolic_leaf_scores, k=k, dim=-1).indices)
    if neural_leaf_scores is not None and neural_k > 0:
        k = min(neural_k, leaf_count)
        candidate_parts.append(torch.topk(neural_leaf_scores, k=k, dim=-1).indices)
    combined_k = min(
        leaf_count,
        max(top_k, top_k * candidate_multiplier, top_k + neural_k + symbolic_k),
    )
    candidate_parts.append(torch.topk(leaf_scores, k=combined_k, dim=-1).indices)
    candidate_order = torch.cat(candidate_parts, dim=-1)
    selected_indices = _first_unique_indices(candidate_order, leaf_count=leaf_count, top_k=top_k)
    selected_scores = leaf_scores.gather(-1, selected_indices)
    return selected_scores, selected_indices


def _gated_rerank_topk_leaves(
    leaf_scores: torch.Tensor,
    neural_leaf_scores: torch.Tensor | None,
    *,
    top_k: int,
    candidate_top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if neural_leaf_scores is None:
        return torch.topk(leaf_scores, k=top_k, dim=-1)
    leaf_count = leaf_scores.shape[-1]
    candidate_top_k = min(max(candidate_top_k or top_k, top_k), leaf_count)
    candidate_indices = torch.topk(neural_leaf_scores, k=candidate_top_k, dim=-1).indices
    candidate_scores = leaf_scores.gather(-1, candidate_indices)
    local = torch.topk(candidate_scores, k=top_k, dim=-1)
    selected_indices = candidate_indices.gather(-1, local.indices)
    selected_scores = leaf_scores.gather(-1, selected_indices)
    return selected_scores, selected_indices


def _rank_fusion_topk_leaves(
    leaf_scores: torch.Tensor,
    neural_leaf_scores: torch.Tensor | None,
    symbolic_leaf_scores: torch.Tensor | None,
    *,
    top_k: int,
    neural_weight: float,
    symbolic_weight: float,
    combined_weight: float,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fusion_scores = torch.zeros_like(leaf_scores)
    if combined_weight:
        fusion_scores = fusion_scores + combined_weight * _reciprocal_rank_scores(
            leaf_scores,
            offset=offset,
        )
    if neural_leaf_scores is not None and neural_weight:
        fusion_scores = fusion_scores + neural_weight * _reciprocal_rank_scores(
            neural_leaf_scores,
            offset=offset,
        )
    if symbolic_leaf_scores is not None and symbolic_weight:
        fusion_scores = fusion_scores + symbolic_weight * _reciprocal_rank_scores(
            symbolic_leaf_scores,
            offset=offset,
        )
    return torch.topk(fusion_scores, k=top_k, dim=-1)


def _reciprocal_rank_scores(scores: torch.Tensor, *, offset: float) -> torch.Tensor:
    sorted_indices = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(scores)
    rank_values = torch.arange(
        1,
        scores.shape[-1] + 1,
        dtype=scores.dtype,
        device=scores.device,
    ).view(*([1] * (scores.ndim - 1)), scores.shape[-1])
    ranks.scatter_(-1, sorted_indices, rank_values.expand_as(sorted_indices))
    return 1.0 / (ranks.to(dtype=scores.dtype) + offset)


def _diverse_topk_leaves(
    leaf_scores: torch.Tensor,
    leaf_summaries: torch.Tensor,
    *,
    top_k: int,
    diversity_weight: float,
    candidate_multiplier: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MMR-style top-k over a high-score candidate pool.

    The first leaf is the highest-scoring leaf. Later leaves trade route score
    against cosine similarity to already selected leaves, which discourages
    reviving several near-duplicate summaries for multi-source tasks.
    """

    leaf_count = leaf_scores.shape[-1]
    candidate_count = min(leaf_count, max(top_k, top_k * candidate_multiplier))
    candidate_scores, candidate_indices = torch.topk(leaf_scores, k=candidate_count, dim=-1)
    leaf_vectors = F.normalize(leaf_summaries.mean(dim=2), dim=-1)
    expanded_vectors = leaf_vectors.unsqueeze(1).expand(
        leaf_vectors.shape[0],
        leaf_scores.shape[1],
        leaf_count,
        leaf_vectors.shape[-1],
    )
    candidate_vectors = expanded_vectors.gather(
        2,
        candidate_indices.unsqueeze(-1).expand(
            *candidate_indices.shape,
            leaf_vectors.shape[-1],
        ),
    )
    selected_mask = torch.zeros_like(candidate_scores, dtype=torch.bool)
    max_similarity = torch.zeros_like(candidate_scores)
    selected_local_indices = []
    min_value = torch.finfo(candidate_scores.dtype).min

    for _ in range(top_k):
        adjusted_scores = candidate_scores - diversity_weight * max_similarity
        adjusted_scores = adjusted_scores.masked_fill(selected_mask, min_value)
        local_indices = adjusted_scores.argmax(dim=-1)
        selected_local_indices.append(local_indices)
        selected_mask.scatter_(-1, local_indices.unsqueeze(-1), True)
        selected_vectors = candidate_vectors.gather(
            2,
            local_indices.view(*local_indices.shape, 1, 1).expand(
                *local_indices.shape,
                1,
                candidate_vectors.shape[-1],
            ),
        ).squeeze(2)
        similarity = torch.einsum("bqch,bqh->bqc", candidate_vectors, selected_vectors)
        max_similarity = torch.maximum(max_similarity, similarity.clamp_min(0.0))

    selected_local = torch.stack(selected_local_indices, dim=-1)
    selected_indices = candidate_indices.gather(-1, selected_local)
    selected_scores = leaf_scores.gather(-1, selected_indices)
    return selected_scores, selected_indices


def _first_unique_indices(candidate_order: torch.Tensor, *, leaf_count: int, top_k: int) -> torch.Tensor:
    selected_indices = torch.zeros(*candidate_order.shape[:2], top_k, dtype=torch.long, device=candidate_order.device)
    selected_count = torch.zeros(*candidate_order.shape[:2], dtype=torch.long, device=candidate_order.device)
    seen = torch.zeros(*candidate_order.shape[:2], leaf_count, dtype=torch.bool, device=candidate_order.device)
    slots = torch.arange(top_k, device=candidate_order.device).view(1, 1, top_k)
    for position in range(candidate_order.shape[-1]):
        candidate = candidate_order[..., position]
        already_seen = seen.gather(-1, candidate.unsqueeze(-1)).squeeze(-1)
        can_take = (~already_seen) & (selected_count < top_k)
        slot = selected_count.clamp(max=max(top_k - 1, 0))
        write_mask = can_take.unsqueeze(-1) & (slots == slot.unsqueeze(-1))
        selected_indices = torch.where(write_mask, candidate.unsqueeze(-1), selected_indices)
        seen_update = torch.zeros_like(seen).scatter(-1, candidate.unsqueeze(-1), can_take.unsqueeze(-1))
        seen = seen | seen_update
        selected_count = selected_count + can_take.to(dtype=selected_count.dtype)
    return selected_indices


def _gate_scores_by_topk(
    scores: torch.Tensor,
    gate_scores: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    leaf_count = scores.shape[-1]
    top_k = min(max(top_k, 1), leaf_count)
    top_indices = torch.topk(gate_scores, k=top_k, dim=-1).indices
    gate_mask = torch.zeros_like(scores, dtype=torch.bool)
    gate_mask.scatter_(-1, top_indices, True)
    return torch.where(gate_mask, scores, torch.zeros_like(scores))


def _normalize_scores(scores: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "none":
        return scores
    if mode == "zscore":
        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (scores - mean) / std
    if mode == "minmax":
        min_score = scores.min(dim=-1, keepdim=True).values
        max_score = scores.max(dim=-1, keepdim=True).values
        return (scores - min_score) / (max_score - min_score).clamp_min(1e-6)
    raise ValueError(f"unknown score normalization mode: {mode}")


def _revived_leaf_mask_from_top_values(
    top_values: torch.Tensor,
    *,
    min_k: int | None,
    margin: float,
) -> torch.Tensor:
    finite = torch.isfinite(top_values)
    if min_k is None:
        return finite
    top_k = top_values.shape[-1]
    min_k = min(min_k, top_k)
    ranks = torch.arange(top_k, device=top_values.device).view(
        *([1] * (top_values.ndim - 1)),
        top_k,
    )
    floor_mask = ranks < min_k
    within_margin = top_values >= top_values[..., :1] - margin
    return finite & (floor_mask | within_margin)


def _apply_adaptive_budget_mask(
    revived_leaf_mask: torch.Tensor,
    leaf_scores: torch.Tensor,
    top_values: torch.Tensor,
    *,
    adaptive_mode: str,
    adaptive_min_k: int | None,
    entropy_threshold: float,
) -> torch.Tensor:
    if adaptive_mode == "none" or adaptive_min_k is None:
        return revived_leaf_mask
    if adaptive_mode != "entropy":
        raise ValueError(f"unknown adaptive budget mode: {adaptive_mode}")
    top_k = top_values.shape[-1]
    min_k = min(adaptive_min_k, top_k)
    ranks = torch.arange(top_k, device=top_values.device).view(
        *([1] * (top_values.ndim - 1)),
        top_k,
    )
    floor_mask = ranks < min_k
    leaf_count = leaf_scores.shape[-1]
    if leaf_count <= 1:
        uncertain = torch.ones_like(leaf_scores[..., :1], dtype=torch.bool)
    else:
        probs = torch.softmax(leaf_scores, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
        normalized_entropy = entropy / math.log(float(leaf_count))
        uncertain = normalized_entropy >= entropy_threshold
    return revived_leaf_mask & (floor_mask | uncertain)


def _causal_window_mean(
    states: torch.Tensor,
    positions: torch.Tensor,
    *,
    window_size: int,
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = states.shape
    positions = positions.clamp(min=0, max=seq_len - 1)
    starts = (positions - window_size + 1).clamp_min(0)
    ends = positions + 1
    prefix = F.pad(states.cumsum(dim=1), (0, 0, 1, 0))
    gather_shape = (*positions.shape, hidden_size)
    start_sums = prefix.gather(1, starts.unsqueeze(-1).expand(gather_shape))
    end_sums = prefix.gather(1, ends.unsqueeze(-1).expand(gather_shape))
    lengths = (ends - starts).clamp_min(1).to(dtype=states.dtype).unsqueeze(-1)
    if start_sums.shape[0] != batch_size:
        raise RuntimeError("unexpected batch shape while pooling query states")
    return (end_sums - start_sums) / lengths


def _gather_positions(states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    positions = positions.clamp(min=0, max=states.shape[1] - 1)
    return states.gather(
        1,
        positions.unsqueeze(-1).expand(*positions.shape, states.shape[-1]),
    )


def _causal_window_states(
    states: torch.Tensor,
    positions: torch.Tensor,
    *,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_size = states.shape
    positions = positions.clamp(min=0, max=seq_len - 1)
    offsets = torch.arange(window_size, device=states.device).view(1, 1, window_size)
    offsets = offsets - (window_size - 1)
    indices = positions.unsqueeze(-1) + offsets
    valid = (indices >= 0) & (indices < seq_len)
    safe_indices = indices.clamp(min=0, max=seq_len - 1)
    expanded_states = states.unsqueeze(1).expand(batch_size, positions.shape[1], seq_len, hidden_size)
    gathered = expanded_states.gather(
        2,
        safe_indices.unsqueeze(-1).expand(*safe_indices.shape, hidden_size),
    )
    return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered)), valid


def _causal_window_token_ids(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    window_size: int,
) -> torch.Tensor:
    batch_size, seq_len = token_ids.shape
    positions = positions.clamp(min=0, max=seq_len - 1)
    offsets = torch.arange(window_size, device=token_ids.device).view(1, 1, window_size)
    offsets = offsets - (window_size - 1)
    indices = positions.unsqueeze(-1) + offsets
    valid = (indices >= 0) & (indices < seq_len)
    safe_indices = indices.clamp(min=0, max=seq_len - 1)
    expanded_tokens = token_ids.unsqueeze(1).expand(batch_size, positions.shape[1], seq_len)
    windows = expanded_tokens.gather(2, safe_indices)
    padding = torch.full_like(windows, -1)
    return torch.where(valid, windows, padding)


def _symbolic_leaf_scores(
    query_token_ids: torch.Tensor,
    leaf_token_ids: torch.Tensor,
    leaf_mask: torch.Tensor,
    *,
    hash_bins: int,
    ngrams: tuple[int, ...],
    weights: tuple[float, ...] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if weights is None:
        weights = tuple(1.0 for _ in ngrams)
    if len(weights) != len(ngrams):
        raise ValueError("weights must match ngrams length")
    query_mask = query_token_ids >= 0
    score = torch.zeros(
        (*query_token_ids.shape[:2], leaf_token_ids.shape[1]),
        dtype=dtype,
        device=query_token_ids.device,
    )
    weight_sum = 0.0
    for ngram, weight in zip(ngrams, weights):
        if weight == 0:
            continue
        query_sketch = _hashed_ngram_sketch(
            query_token_ids,
            query_mask,
            hash_bins=hash_bins,
            ngram=ngram,
            dtype=dtype,
        )
        leaf_sketch = _hashed_ngram_sketch(
            leaf_token_ids,
            leaf_mask,
            hash_bins=hash_bins,
            ngram=ngram,
            dtype=dtype,
        )
        score = score + float(weight) * torch.einsum("bqh,blh->bql", query_sketch, leaf_sketch)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return score
    return score / weight_sum


def _hashed_ngram_sketch(
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    hash_bins: int,
    ngram: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    base_shape = token_ids.shape[:-1]
    length = token_ids.shape[-1]
    sketch = torch.zeros(
        *base_shape,
        hash_bins,
        dtype=dtype,
        device=token_ids.device,
    )
    if length < ngram:
        return sketch
    span = length - ngram + 1
    safe_ids = token_ids.clamp(min=0).to(dtype=torch.long)
    valid = mask[..., :span].clone()
    hashed = torch.zeros(*base_shape, span, dtype=torch.long, device=token_ids.device)
    for offset in range(ngram):
        valid = valid & mask[..., offset : offset + span]
        hashed = (hashed * 257 + safe_ids[..., offset : offset + span] + 1) % hash_bins
    sketch.scatter_add_(-1, hashed, valid.to(dtype=dtype))
    norm = sketch.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return sketch / norm


def _candidate_indices_from_recent_and_leaves(
    *,
    seq_len: int,
    recent_start: int,
    leaf_token_indices: torch.Tensor,
    leaf_token_mask: torch.Tensor,
    revived_leaf_indices: torch.Tensor,
    revived_leaf_mask: torch.Tensor,
    batch_size: int,
    query_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    recent_indices = torch.arange(recent_start, seq_len, device=device, dtype=torch.long)
    recent_indices = recent_indices.view(1, 1, -1).expand(batch_size, query_count, -1)
    recent_mask = torch.ones_like(recent_indices, dtype=torch.bool)
    if revived_leaf_indices.numel() == 0:
        return recent_indices, recent_mask

    batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    batch_index = batch_index.expand_as(revived_leaf_indices)
    safe_leaf_indices = revived_leaf_indices.clamp(min=0)
    revived_token_indices = leaf_token_indices[safe_leaf_indices]
    revived_token_mask = leaf_token_mask[batch_index, safe_leaf_indices]
    revived_token_mask = revived_token_mask & revived_leaf_mask.unsqueeze(-1)
    revived_token_indices = revived_token_indices.reshape(batch_size, query_count, -1)
    revived_token_mask = revived_token_mask.reshape(batch_size, query_count, -1)
    candidate_indices = torch.cat([revived_token_indices, recent_indices], dim=-1)
    candidate_mask = torch.cat([revived_token_mask, recent_mask], dim=-1)
    return candidate_indices, candidate_mask
