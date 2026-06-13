from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla import UABLAConfig
from uabla.routing import (
    _expand_candidate_spans,
    _mask_duplicate_candidates,
    build_candidate_indices,
    build_candidate_indices_vectorized,
    compute_centroid_route_scores,
    compute_superblock_route_scores,
    summarize_block_centroids,
    summarize_superblock_centroids,
    uncertainty_to_bucket_values,
)


def test_uncertainty_buckets_are_monotonic() -> None:
    uncertainty = torch.tensor([-5.0, -1.5, 3.0])

    values = uncertainty_to_bucket_values(
        uncertainty,
        (4, 8, 16, 32),
        beta=-1.5,
        gamma=0.5,
    )

    assert values.tolist()[0] <= values.tolist()[1] <= values.tolist()[2]


def test_four_centroid_summary_shapes() -> None:
    config = UABLAConfig(hidden_size=16, routing_dim=8, value_dim=12, position_dim=4, block_size=4)
    mu = torch.randn(2, 9, config.routing_dim)
    log_sigma = torch.zeros_like(mu)
    assignment_logits = torch.randn(2, 9, config.centroids_per_block)

    centroids = summarize_block_centroids(
        mu,
        log_sigma,
        assignment_logits,
        block_size=config.block_size,
    )

    assert centroids.mu.shape == (2, 3, 4, 8)
    assert centroids.log_sigma.shape == (2, 3, 4, 8)
    assert centroids.weight_sum.shape == (2, 3, 4)


def test_candidate_indices_are_causal() -> None:
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=2,
        centroid_hit_buckets=(4,),
        token_k_buckets=(4,),
        adaptive_budgets=False,
    )
    mu = torch.randn(1, 10, config.routing_dim)
    log_sigma = torch.zeros_like(mu)
    assignment_logits = torch.randn(1, 10, config.centroids_per_block)
    centroids = summarize_block_centroids(
        mu,
        log_sigma,
        assignment_logits,
        block_size=config.block_size,
    )
    route_scores, routeable_mask = compute_centroid_route_scores(mu, log_sigma, centroids, config)

    candidates = build_candidate_indices(
        mu,
        log_sigma,
        centroids,
        config,
        route_scores=route_scores,
        routeable_mask=routeable_mask,
    )

    for token_idx in range(mu.shape[1]):
        valid = candidates.indices[0, token_idx][candidates.mask[0, token_idx]]
        assert valid.numel() > 0
        assert int(valid.max().item()) <= token_idx


def test_vectorized_candidate_indices_are_causal_and_fixed_shape() -> None:
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=2,
        centroid_hit_buckets=(4,),
        token_k_buckets=(4,),
        superblock_size_blocks=2,
        superblock_hit_buckets=(2,),
        adaptive_budgets=False,
    )
    mu = torch.randn(2, 10, config.routing_dim)
    log_sigma = torch.zeros_like(mu)
    assignment_logits = torch.randn(2, 10, config.centroids_per_block)
    centroids = summarize_block_centroids(
        mu,
        log_sigma,
        assignment_logits,
        block_size=config.block_size,
    )
    super_centroids = summarize_superblock_centroids(
        centroids,
        superblock_size_blocks=config.superblock_size_blocks,
    )
    route_scores, routeable_mask = compute_centroid_route_scores(mu, log_sigma, centroids, config)
    super_scores, super_mask = compute_superblock_route_scores(
        mu,
        log_sigma,
        super_centroids,
        config,
    )

    candidates = build_candidate_indices_vectorized(
        mu,
        log_sigma,
        centroids,
        config,
        route_scores=route_scores,
        routeable_mask=routeable_mask,
        superblock_route_scores=super_scores,
        superblock_routeable_mask=super_mask,
    )

    assert candidates.indices.shape == (
        2,
        10,
        config.local_window + max(config.centroid_hit_buckets) * config.block_size,
    )
    for batch_idx in range(mu.shape[0]):
        for token_idx in range(mu.shape[1]):
            valid = candidates.indices[batch_idx, token_idx][candidates.mask[batch_idx, token_idx]]
            assert valid.numel() > 0
            assert int(valid.max().item()) <= token_idx


def test_duplicate_candidate_mask_keeps_first_occurrence() -> None:
    indices = torch.tensor([[[0, 1, 1, 2, 0, 3, -1]]])
    mask = torch.tensor([[[True, True, True, True, True, True, False]]])

    _, deduped = _mask_duplicate_candidates(indices, mask)

    assert deduped.tolist() == [[[True, True, False, True, False, True, False]]]


def test_candidate_span_expansion_is_causal() -> None:
    indices = torch.tensor([[[0, 0], [0, 1], [1, 2]]])
    mask = torch.tensor([[[True, True], [True, True], [True, True]]])

    expanded, expanded_mask = _expand_candidate_spans(
        indices,
        mask,
        seq_len=8,
        left=1,
        right=2,
    )

    assert expanded.shape[-1] == indices.shape[-1] * 4
    for token_idx in range(expanded.shape[1]):
        valid = expanded[0, token_idx][expanded_mask[0, token_idx]]
        assert int(valid.max().item()) <= token_idx


def test_route_scores_are_block_level_not_token_pair_level() -> None:
    config = UABLAConfig(hidden_size=16, routing_dim=8, value_dim=12, position_dim=4, block_size=4)
    mu = torch.randn(2, 10, config.routing_dim)
    log_sigma = torch.zeros_like(mu)
    assignment_logits = torch.randn(2, 10, config.centroids_per_block)
    centroids = summarize_block_centroids(
        mu,
        log_sigma,
        assignment_logits,
        block_size=config.block_size,
    )

    route_scores, routeable_mask = compute_centroid_route_scores(mu, log_sigma, centroids, config)

    assert route_scores.shape == (2, 10, 3, 4)
    assert routeable_mask.shape == (2, 10, 3, 4)


def test_multiscale_superblock_shapes() -> None:
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        superblock_size_blocks=2,
    )
    mu = torch.randn(2, 18, config.routing_dim)
    log_sigma = torch.zeros_like(mu)
    assignment_logits = torch.randn(2, 18, config.centroids_per_block)
    centroids = summarize_block_centroids(
        mu,
        log_sigma,
        assignment_logits,
        block_size=config.block_size,
    )

    super_centroids = summarize_superblock_centroids(
        centroids,
        superblock_size_blocks=config.superblock_size_blocks,
    )
    super_scores, super_routeable = compute_superblock_route_scores(
        mu,
        log_sigma,
        super_centroids,
        config,
    )

    assert centroids.mu.shape == (2, 5, 4, 8)
    assert super_centroids.mu.shape == (2, 3, 4, 8)
    assert super_scores.shape == (2, 18, 3, 4)
    assert super_routeable.shape == (2, 18, 3, 4)
