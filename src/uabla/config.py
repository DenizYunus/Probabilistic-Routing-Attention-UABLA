"""Configuration for the locked UABLA-448 V1 design."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UABLAConfig:
    """UABLA attention configuration.

    Defaults follow docs/UABLA_V1_LOCKED.md. Tests and toy models can override
    the dimensions while keeping the same routing structure.
    """

    hidden_size: int
    routing_dim: int = 64
    value_dim: int = 256
    position_dim: int = 64
    block_size: int = 128
    centroids_per_block: int = 4
    local_window: int = 256
    vectorized_routing: bool = True
    use_multiscale_routing: bool = True
    superblock_size_blocks: int = 8
    superblock_hit_buckets: tuple[int, ...] = (2, 4, 8, 16)
    centroid_hit_buckets: tuple[int, ...] = (4, 8, 16, 32)
    token_k_buckets: tuple[int, ...] = (16, 32, 64, 128)
    alpha: float = 0.1
    tau: float = 1.0
    min_log_sigma: float = -5.0
    max_log_sigma: float = 2.0
    eps: float = 1e-6
    beta: float = -1.5
    gamma: float = 0.5
    max_position_buckets: int = 64
    dropout: float = 0.0
    adaptive_budgets: bool = True
    use_memory_importance: bool = True
    memory_importance_score_scale: float = 0.25
    hybrid_score_rank: int = 16
    hybrid_score_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.routing_dim <= 0:
            raise ValueError("routing_dim must be positive")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if self.position_dim <= 0:
            raise ValueError("position_dim must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.centroids_per_block != 4:
            raise ValueError("UABLA V1 is locked to exactly 4 centroids per block")
        if self.local_window <= 0:
            raise ValueError("local_window must be positive")
        if self.superblock_size_blocks <= 0:
            raise ValueError("superblock_size_blocks must be positive")
        if not self.superblock_hit_buckets:
            raise ValueError("superblock_hit_buckets cannot be empty")
        if not self.centroid_hit_buckets:
            raise ValueError("centroid_hit_buckets cannot be empty")
        if not self.token_k_buckets:
            raise ValueError("token_k_buckets cannot be empty")
        if any(k <= 0 for k in self.superblock_hit_buckets):
            raise ValueError("superblock_hit_buckets must be positive")
        if any(k <= 0 for k in self.centroid_hit_buckets):
            raise ValueError("centroid_hit_buckets must be positive")
        if any(k <= 0 for k in self.token_k_buckets):
            raise ValueError("token_k_buckets must be positive")
        if tuple(sorted(self.superblock_hit_buckets)) != self.superblock_hit_buckets:
            raise ValueError("superblock_hit_buckets must be sorted ascending")
        if tuple(sorted(self.centroid_hit_buckets)) != self.centroid_hit_buckets:
            raise ValueError("centroid_hit_buckets must be sorted ascending")
        if tuple(sorted(self.token_k_buckets)) != self.token_k_buckets:
            raise ValueError("token_k_buckets must be sorted ascending")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.max_position_buckets <= 0:
            raise ValueError("max_position_buckets must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.memory_importance_score_scale < 0:
            raise ValueError("memory_importance_score_scale cannot be negative")
        if self.hybrid_score_rank < 0:
            raise ValueError("hybrid_score_rank cannot be negative")
        if self.hybrid_score_scale < 0:
            raise ValueError("hybrid_score_scale cannot be negative")

    @property
    def cache_dim(self) -> int:
        return self.routing_dim * 2 + self.value_dim + self.position_dim
