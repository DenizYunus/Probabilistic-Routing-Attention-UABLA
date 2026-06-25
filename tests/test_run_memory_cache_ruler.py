from __future__ import annotations

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from scripts.run_memory_cache_ruler import (
    answer_source_set_indices,
    budget_ladder_loss,
    collect_metrics_for_mode,
    neural_gate_loss,
    source_indices_for_mode,
    source_losses,
)
from uabla.memory_cache import RevivalCacheOutput


def test_dual_source_target_mode_keeps_per_token_targets() -> None:
    batch = {
        "answer_index": torch.tensor([[12, 13]]),
        "answer_source_index": torch.tensor([[1, 9]]),
    }

    assert source_indices_for_mode(batch, "dual").tolist() == [[1, 9]]
    assert source_indices_for_mode(batch, "answer-set").tolist() == [[[1, 9], [1, 9]]]
    assert answer_source_set_indices(batch["answer_source_index"]).tolist() == [[[1, 9], [1, 9]]]


def test_dual_source_losses_add_answer_set_coverage() -> None:
    output = _dummy_dual_output()
    source_indices = torch.tensor([[1, 9]])
    args = Namespace(
        source_target_mode="dual",
        leaf_size=4,
        revived_top_k=2,
        survival_margin=0.0,
        survival_temperature=1.0,
        source_coverage_temperature=1.0,
        seq_len=16,
    )

    ce_loss, survival_loss, coverage_loss = source_losses(output, source_indices, args)
    metrics = collect_metrics_for_mode(output, source_indices, args)

    assert torch.isfinite(ce_loss)
    assert torch.isfinite(survival_loss)
    assert torch.isfinite(coverage_loss)
    assert coverage_loss.item() > 0.0
    assert metrics.source_hit_rate == 1.0
    assert metrics.source_set_hit_rate == 1.0
    assert metrics.source_coverage_loss is not None


def test_budget_ladder_loss_averages_multiple_survival_budgets() -> None:
    output = _dummy_dual_output()
    source_indices = torch.tensor([[1, 9]])
    args = Namespace(
        source_target_mode="dual",
        leaf_size=4,
        budget_ladder_weight=0.5,
        budget_ladder_top_ks=(1, 2),
        survival_margin=0.0,
        survival_temperature=1.0,
    )

    loss = budget_ladder_loss(output, source_indices, args)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_neural_gate_loss_uses_neural_scores() -> None:
    output = _dummy_dual_output()
    source_indices = torch.tensor([[1, 9]])
    args = Namespace(
        source_target_mode="dual",
        leaf_size=4,
        revived_top_k=2,
        symbolic_gate_top_k=2,
        neural_gate_top_k=None,
        neural_gate_loss_weight=0.5,
        survival_margin=0.0,
        survival_temperature=1.0,
    )

    loss = neural_gate_loss(output, source_indices, args)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def _dummy_dual_output() -> RevivalCacheOutput:
    candidate_indices = torch.tensor(
        [
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [0, 1, 2, 3, 8, 9, 10, 11],
            ]
        ]
    )
    candidate_mask = torch.ones_like(candidate_indices, dtype=torch.bool)
    leaf_scores = torch.tensor([[[4.0, 0.0, 3.0], [3.0, 0.0, 4.0]]])
    return RevivalCacheOutput(
        candidate_indices=candidate_indices,
        candidate_mask=candidate_mask,
        revived_leaf_indices=torch.tensor([[[0, 2], [2, 0]]]),
        revived_leaf_mask=torch.tensor([[[True, True], [True, True]]]),
        leaf_scores=leaf_scores,
        leaf_summaries=torch.zeros(1, 3, 1, 2),
        leaf_token_indices=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
        leaf_token_mask=torch.ones(1, 3, 4, dtype=torch.bool),
        recent_start=12,
        neural_leaf_scores=leaf_scores - 0.1,
    )
