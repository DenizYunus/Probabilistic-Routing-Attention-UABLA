from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla import UABLAAttention, UABLAConfig
from uabla.losses import (
    direct_routing_supervision_loss,
    guarded_budget_loss,
    route_entropy_floor_loss,
    routing_diagnostics,
    routing_distillation_loss,
    token_contrastive_retrieval_loss,
)


def test_routing_distillation_loss_is_differentiable() -> None:
    torch.manual_seed(3)
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=3,
        centroid_hit_buckets=(4, 8),
        token_k_buckets=(2, 4),
    )
    module = UABLAAttention(config)
    x = torch.randn(1, 12, config.hidden_size)
    result = module(x, return_routing=True)
    teacher = torch.softmax(torch.randn(1, 12, 12), dim=-1)

    assert result.route_scores is not None
    assert result.routeable_mask is not None
    loss = routing_distillation_loss(
        result.route_scores,
        result.routeable_mask,
        teacher,
        region_size_tokens=config.block_size,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert module.assign.weight.grad is not None
    assert module.assign.weight.grad.abs().sum().item() > 0.0


def test_guarded_budget_loss_and_diagnostics_are_finite() -> None:
    torch.manual_seed(4)
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=3,
        centroid_hit_buckets=(4, 8),
        token_k_buckets=(2, 4),
    )
    module = UABLAAttention(config)
    x = torch.randn(1, 12, config.hidden_size)
    result = module(x, return_routing=True, return_attention=True)

    assert result.candidates is not None
    assert result.route_scores is not None
    assert result.routeable_mask is not None
    logits = torch.randn(1, 12, 32)
    budget_loss = guarded_budget_loss(result.candidates.token_budgets, logits=logits)
    diagnostics = routing_diagnostics(
        result.candidates,
        result.route_scores,
        result.routeable_mask,
        attention=result.attention,
    )

    assert torch.isfinite(budget_loss)
    assert diagnostics
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_route_entropy_floor_loss_penalizes_collapsed_routes() -> None:
    route_scores = torch.tensor([[[[10.0], [0.0], [0.0], [0.0]]]], requires_grad=True)
    routeable_mask = torch.ones_like(route_scores, dtype=torch.bool)

    loss = route_entropy_floor_loss(
        route_scores,
        routeable_mask,
        min_normalized_entropy=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert route_scores.grad is not None
    assert route_scores.grad.abs().sum().item() > 0


def test_direct_routing_supervision_loss_is_differentiable() -> None:
    torch.manual_seed(6)
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=3,
        centroid_hit_buckets=(4, 8),
        token_k_buckets=(2, 4),
    )
    module = UABLAAttention(config)
    x = torch.randn(1, 16, config.hidden_size)
    result = module(x, return_routing=True)

    assert result.route_scores is not None
    assert result.routeable_mask is not None
    loss = direct_routing_supervision_loss(
        result.route_scores,
        result.routeable_mask,
        answer_indices=torch.tensor([[14]]),
        source_indices=torch.tensor([[5]]),
        region_size_tokens=config.block_size,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert module.assign.weight.grad is not None
    assert module.assign.weight.grad.abs().sum().item() > 0.0


def test_token_contrastive_retrieval_loss_is_differentiable() -> None:
    torch.manual_seed(7)
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=8,
        centroid_hit_buckets=(4, 8),
        token_k_buckets=(2, 4),
    )
    module = UABLAAttention(config)
    x = torch.randn(1, 16, config.hidden_size)
    result = module(x, return_routing=True, return_token_scores=True)

    assert result.candidates is not None
    assert result.token_scores is not None
    loss = token_contrastive_retrieval_loss(
        result.token_scores,
        result.candidates.indices,
        result.candidates.mask,
        answer_indices=torch.tensor([[14]]),
        source_indices=torch.tensor([[13]]),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert module.q_mu.weight.grad is not None
    assert module.q_mu.weight.grad.abs().sum().item() > 0.0


def test_distillation_returns_zero_when_no_regions_are_routeable() -> None:
    route_scores = torch.full((1, 4, 2, 4), torch.finfo(torch.float32).min)
    routeable_mask = torch.zeros_like(route_scores, dtype=torch.bool)
    teacher = torch.softmax(torch.randn(1, 4, 4), dim=-1)

    loss = routing_distillation_loss(
        route_scores,
        routeable_mask,
        teacher,
        region_size_tokens=4,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_token_contrastive_retrieval_returns_zero_when_source_is_not_candidate() -> None:
    token_scores = torch.randn(1, 4, 3)
    candidate_indices = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 1, 2], [1, 2, 3]]])
    candidate_mask = torch.ones_like(candidate_indices, dtype=torch.bool)

    loss = token_contrastive_retrieval_loss(
        token_scores,
        candidate_indices,
        candidate_mask,
        answer_indices=torch.tensor([[3]]),
        source_indices=torch.tensor([[0]]),
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_direct_routing_supervision_returns_zero_when_target_is_not_routeable() -> None:
    route_scores = torch.full((1, 4, 2, 4), torch.finfo(torch.float32).min)
    routeable_mask = torch.zeros_like(route_scores, dtype=torch.bool)

    loss = direct_routing_supervision_loss(
        route_scores,
        routeable_mask,
        answer_indices=torch.tensor([[3]]),
        source_indices=torch.tensor([[1]]),
        region_size_tokens=4,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
