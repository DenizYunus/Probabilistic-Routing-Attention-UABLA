from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla import UABLAAttention, UABLAConfig, UABLAOutput


def tiny_config() -> UABLAConfig:
    return UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=3,
        centroid_hit_buckets=(4, 8),
        token_k_buckets=(2, 4),
    )


def test_attention_output_and_cache_shapes() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    module = UABLAAttention(config)
    x = torch.randn(2, 10, config.hidden_size)

    result = module(x, return_cache=True, return_routing=True)

    assert isinstance(result, UABLAOutput)
    assert result.output.shape == x.shape
    assert result.cache is not None
    assert result.cache.shape == (2, 10, config.cache_dim)
    assert result.centroids is not None
    assert result.centroids.mu.shape == (2, 3, 4, config.routing_dim)
    assert result.superblock_centroids is not None
    assert result.superblock_centroids.mu.shape == (2, 1, 4, config.routing_dim)
    assert result.shifted_centroids is not None
    assert result.shifted_centroids.mu.shape == (2, 2, 4, config.routing_dim)
    assert result.candidates is not None
    assert result.candidates.indices.shape[:2] == (2, 10)
    assert result.route_scores is not None
    assert result.route_scores.shape == (2, 10, 3, 4)
    assert result.shifted_route_scores is not None
    assert result.shifted_route_scores.shape == (2, 10, 2, 4)
    assert result.memory_importance is not None
    assert result.memory_importance.shape == (2, 10)


def test_attention_is_causal_for_prefix_outputs() -> None:
    torch.manual_seed(1)
    config = tiny_config()
    module = UABLAAttention(config)
    module.eval()
    x = torch.randn(1, 10, config.hidden_size)
    changed_future = x.clone()
    changed_future[:, 6:] = torch.randn_like(changed_future[:, 6:])

    with torch.no_grad():
        out_a = module(x)
        out_b = module(changed_future)

    assert torch.allclose(out_a[:, :6], out_b[:, :6], atol=1e-5, rtol=1e-5)


def test_straight_through_router_gets_assignment_gradients() -> None:
    torch.manual_seed(2)
    config = tiny_config()
    module = UABLAAttention(config)
    x = torch.randn(1, 10, config.hidden_size)

    result = module(x)
    loss = result.square().mean()
    loss.backward()

    assert module.assign.weight.grad is not None
    assert module.assign.weight.grad.abs().sum().item() > 0.0
