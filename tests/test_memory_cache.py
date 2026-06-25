from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla.memory_cache import (
    AverageLeafCompressor,
    HybridLeafCompressor,
    MemoryCacheRouteProbe,
    RevivalCacheConfig,
    RevivalCacheOutput,
    RevivalMemoryCache,
    UsefulnessLedger,
    collect_probe_metrics,
    collect_probe_set_metrics,
    memory_cache_source_hit_rate,
    memory_cache_source_leaf_accuracy,
    memory_cache_source_leaf_loss,
    memory_cache_source_set_coverage_loss,
    memory_cache_source_set_leaf_accuracy,
    memory_cache_source_set_leaf_loss,
    memory_cache_source_set_leaf_recall,
    memory_cache_source_set_survival_loss,
    memory_cache_revived_leaf_diversity_loss,
    memory_cache_summary_diversity_loss,
    memory_cache_topk_survival_loss,
    scheduled_recent_window,
)


def test_revival_cache_keeps_recent_and_revives_selected_leaf() -> None:
    hidden = torch.zeros(1, 12, 4)
    hidden[:, 0:4, 0] = 0.1
    hidden[:, 4:8, 0] = 3.0
    query = torch.zeros(1, 1, 4)
    query[:, :, 0] = 1.0
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=4,
            summary_slots=2,
            revived_top_k=1,
        ),
    )
    with torch.no_grad():
        cache.compressor.slot_queries.zero_()
        cache.compressor.slot_queries[:, 0] = 1.0

    output = cache(hidden, query)

    assert output.recent_start == 8
    assert output.revived_leaf_indices.tolist() == [[[1]]]
    assert output.candidate_mask.tolist() == [[[True] * 8]]
    assert output.candidate_indices.tolist() == [[[4, 5, 6, 7, 8, 9, 10, 11]]]
    assert output.active_exact_tokens_per_query.tolist() == [[8]]


def test_source_hit_metrics_detect_revived_source() -> None:
    hidden = torch.zeros(1, 12, 4)
    hidden[:, 4:8, 0] = 3.0
    query = torch.zeros(1, 1, 4)
    query[:, :, 0] = 1.0
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=4,
            summary_slots=2,
            revived_top_k=1,
        ),
    )
    with torch.no_grad():
        cache.compressor.slot_queries.zero_()
        cache.compressor.slot_queries[:, 0] = 1.0
    output = cache(hidden, query)
    source = torch.tensor([[5]])

    assert memory_cache_source_hit_rate(output, source).item() == 1.0
    assert memory_cache_source_leaf_accuracy(output, source, leaf_size=4).item() == 1.0
    assert memory_cache_source_leaf_loss(output, source, leaf_size=4).item() < 0.4
    assert memory_cache_topk_survival_loss(
        output,
        source,
        leaf_size=4,
        top_k=1,
    ).item() < 0.4
    metrics = collect_probe_metrics(output, source, seq_len=12, leaf_size=4)
    assert metrics.source_hit_rate == 1.0
    assert metrics.source_survival_loss < 0.4
    assert metrics.active_token_compression_ratio == 1.5


def test_revival_cache_uses_only_recent_when_sequence_is_short() -> None:
    hidden = torch.randn(2, 5, 8)
    query = torch.randn(2, 3, 8)
    cache = RevivalMemoryCache(
        hidden_size=8,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=16,
            summary_slots=2,
            revived_top_k=2,
        ),
    )

    output = cache(hidden, query)

    assert output.recent_start == 0
    assert output.revived_leaf_indices.shape == (2, 3, 0)
    assert output.candidate_indices.shape == (2, 3, 5)
    assert output.candidate_mask.all()


def test_logsumexp_slot_reduction_runs_and_keeps_shape() -> None:
    hidden = torch.randn(2, 32, 8)
    query = torch.randn(2, 3, 8)
    cache = RevivalMemoryCache(
        hidden_size=8,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=8,
            summary_slots=3,
            revived_top_k=2,
            slot_score_reduction="logsumexp",
            slot_score_temperature=0.5,
        ),
    )

    output = cache(hidden, query)

    assert output.leaf_scores.shape == (2, 3, 6)
    assert output.revived_leaf_indices.shape == (2, 3, 2)


def test_diverse_revival_discourages_near_duplicate_leaves() -> None:
    hidden = torch.tensor(
        [
            [
                [2.0, 0.0],
                [2.0, 0.0],
                [1.9, 0.01],
                [1.9, 0.01],
                [1.0, 5.0],
                [1.0, 5.0],
            ]
        ]
    )
    query = torch.tensor([[[1.0, 0.0]]])
    topk_cache = RevivalMemoryCache(
        hidden_size=2,
        config=RevivalCacheConfig(
            leaf_size=2,
            recent_window=0,
            summary_slots=1,
            revived_top_k=2,
        ),
        compressor=AverageLeafCompressor(hidden_size=2),
    )
    diverse_cache = RevivalMemoryCache(
        hidden_size=2,
        config=RevivalCacheConfig(
            leaf_size=2,
            recent_window=0,
            summary_slots=1,
            revived_top_k=2,
            revival_selection="diverse",
            diversity_weight=1.0,
            diversity_candidate_multiplier=2,
        ),
        compressor=AverageLeafCompressor(hidden_size=2),
    )

    topk_output = topk_cache(hidden, query)
    diverse_output = diverse_cache(hidden, query)

    assert topk_output.revived_leaf_indices.tolist() == [[[0, 1]]]
    assert diverse_output.revived_leaf_indices.tolist() == [[[0, 2]]]


def test_entropy_adaptive_budget_opens_only_uncertain_routes() -> None:
    hidden = torch.zeros(1, 8, 2)
    hidden[:, 0:2, 0] = 5.0
    query = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    cache = RevivalMemoryCache(
        hidden_size=2,
        config=RevivalCacheConfig(
            leaf_size=2,
            recent_window=0,
            summary_slots=1,
            revived_top_k=3,
            adaptive_budget_mode="entropy",
            adaptive_min_k=1,
            adaptive_entropy_threshold=0.95,
        ),
        compressor=AverageLeafCompressor(hidden_size=2),
    )

    output = cache(hidden, query)

    assert output.revived_leaf_mask.tolist() == [[[True, False, False], [True, True, True]]]


def test_channel_mix_revival_runs_without_symbolic_scores() -> None:
    hidden = torch.randn(1, 12, 4)
    query = torch.randn(1, 2, 4)
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=2,
            recent_window=0,
            summary_slots=1,
            revived_top_k=3,
            revival_selection="channel-mix",
            channel_mix_neural_k=1,
            channel_mix_symbolic_k=1,
        ),
        compressor=AverageLeafCompressor(hidden_size=4),
    )

    output = cache(hidden, query)

    assert output.revived_leaf_indices.shape == (1, 2, 3)
    assert output.revived_leaf_mask.all()


def test_symbolic_gate_limits_symbolic_boost_to_neural_candidates() -> None:
    hidden = torch.zeros(1, 8, 2)
    hidden[:, 0:4, 0] = 2.0
    query = torch.tensor([[[1.0, 0.0]]])
    token_ids = torch.tensor([[65, 65, 65, 65, 88, 89, 90, 90]])
    query_token_ids = torch.tensor([[[88, 89, 90, 90]]])
    cache = RevivalMemoryCache(
        hidden_size=2,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=0,
            summary_slots=1,
            revived_top_k=1,
            symbolic_weight=10.0,
            symbolic_hash_bins=64,
            symbolic_ngrams=(2, 3),
            symbolic_gate_top_k=1,
        ),
        compressor=AverageLeafCompressor(hidden_size=2),
    )

    output = cache(
        hidden,
        query,
        token_ids=token_ids,
        query_token_ids=query_token_ids,
    )

    assert output.revived_leaf_indices.tolist() == [[[0]]]
    assert output.symbolic_leaf_scores is not None
    assert output.symbolic_leaf_scores[..., 1].item() == 0.0


def test_rank_fusion_can_select_symbolic_rank_over_combined_score() -> None:
    hidden = torch.zeros(1, 8, 2)
    hidden[:, 0:4, 0] = 2.0
    query = torch.tensor([[[1.0, 0.0]]])
    token_ids = torch.tensor([[65, 65, 65, 65, 88, 89, 90, 90]])
    query_token_ids = torch.tensor([[[88, 89, 90, 90]]])
    cache = RevivalMemoryCache(
        hidden_size=2,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=0,
            summary_slots=1,
            revived_top_k=1,
            symbolic_weight=0.01,
            symbolic_hash_bins=64,
            symbolic_ngrams=(2, 3),
            revival_selection="rank-fusion",
            rank_fusion_neural_weight=0.0,
            rank_fusion_symbolic_weight=1.0,
            rank_fusion_combined_weight=0.0,
        ),
        compressor=AverageLeafCompressor(hidden_size=2),
    )

    output = cache(
        hidden,
        query,
        token_ids=token_ids,
        query_token_ids=query_token_ids,
    )

    assert output.revived_leaf_indices.tolist() == [[[1]]]


def test_gated_rerank_restricts_selection_to_neural_pool() -> None:
    hidden = torch.zeros(1, 8, 2)
    hidden[:, 0:4, 0] = 2.0
    query = torch.tensor([[[1.0, 0.0]]])
    token_ids = torch.tensor([[65, 65, 65, 65, 88, 89, 90, 90]])
    query_token_ids = torch.tensor([[[88, 89, 90, 90]]])
    cache = RevivalMemoryCache(
        hidden_size=2,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=0,
            summary_slots=1,
            revived_top_k=1,
            symbolic_weight=10.0,
            symbolic_hash_bins=64,
            symbolic_ngrams=(2, 3),
            revival_selection="gated-rerank",
            gated_rerank_top_k=1,
        ),
        compressor=AverageLeafCompressor(hidden_size=2),
    )

    output = cache(
        hidden,
        query,
        token_ids=token_ids,
        query_token_ids=query_token_ids,
    )

    assert output.revived_leaf_indices.tolist() == [[[0]]]


def test_symbolic_score_normalization_runs_with_gate() -> None:
    hidden = torch.randn(1, 16, 4)
    query = torch.randn(1, 2, 4)
    token_ids = torch.randint(0, 128, (1, 16))
    query_token_ids = torch.randint(0, 128, (1, 2, 8))
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=0,
            summary_slots=1,
            revived_top_k=2,
            symbolic_weight=1.0,
            symbolic_hash_bins=64,
            symbolic_ngrams=(2, 3),
            symbolic_gate_top_k=2,
            symbolic_score_norm="zscore",
        ),
        compressor=AverageLeafCompressor(hidden_size=4),
    )

    output = cache(
        hidden,
        query,
        token_ids=token_ids,
        query_token_ids=query_token_ids,
    )

    assert output.revived_leaf_indices.shape == (1, 2, 2)
    assert output.symbolic_leaf_scores is not None


def test_address_and_symbolic_context_windows_run() -> None:
    hidden = torch.randn(1, 24, 4)
    query = torch.randn(1, 2, 4)
    token_ids = torch.randint(0, 128, (1, 24))
    query_token_ids = torch.randint(0, 128, (1, 2, 8))
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=0,
            summary_slots=1,
            address_context_left=4,
            address_context_right=2,
            revived_top_k=2,
            symbolic_weight=1.0,
            symbolic_hash_bins=64,
            symbolic_ngrams=(2, 3),
            symbolic_context_left=4,
            symbolic_context_right=2,
        ),
        compressor=AverageLeafCompressor(hidden_size=4),
    )

    output = cache(
        hidden,
        query,
        token_ids=token_ids,
        query_token_ids=query_token_ids,
    )

    assert output.leaf_summaries.shape == (1, 6, 1, 4)
    assert output.leaf_token_indices.shape == (6, 4)
    assert output.candidate_indices.shape[:2] == (1, 2)


def test_revived_leaf_diversity_loss_penalizes_duplicate_summaries() -> None:
    duplicate = RevivalCacheOutput(
        candidate_indices=torch.tensor([[[0, 1, 2, 3]]]),
        candidate_mask=torch.ones(1, 1, 4, dtype=torch.bool),
        revived_leaf_indices=torch.tensor([[[0, 1]]]),
        revived_leaf_mask=torch.tensor([[[True, True]]]),
        leaf_scores=torch.tensor([[[2.0, 1.0]]]),
        leaf_summaries=torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]]),
        leaf_token_indices=torch.tensor([[0, 1], [2, 3]]),
        leaf_token_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        recent_start=4,
    )
    diverse = RevivalCacheOutput(
        candidate_indices=duplicate.candidate_indices,
        candidate_mask=duplicate.candidate_mask,
        revived_leaf_indices=duplicate.revived_leaf_indices,
        revived_leaf_mask=duplicate.revived_leaf_mask,
        leaf_scores=duplicate.leaf_scores,
        leaf_summaries=torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
        leaf_token_indices=duplicate.leaf_token_indices,
        leaf_token_mask=duplicate.leaf_token_mask,
        recent_start=4,
    )

    assert memory_cache_revived_leaf_diversity_loss(duplicate) > memory_cache_revived_leaf_diversity_loss(diverse)


def test_symbolic_router_revives_matching_byte_leaf() -> None:
    hidden = torch.zeros(1, 10, 1)
    query = torch.zeros(1, 1, 1)
    token_ids = torch.tensor([[65, 65, 65, 65, 88, 89, 90, 90, 10, 11]])
    query_token_ids = torch.tensor([[[88, 89, 90, 90]]])
    cache = RevivalMemoryCache(
        hidden_size=1,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=2,
            summary_slots=1,
            revived_top_k=1,
            symbolic_weight=5.0,
            symbolic_hash_bins=64,
            symbolic_ngram=2,
            symbolic_ngrams=(2, 3),
            symbolic_ngram_weights=(0.5, 1.0),
        ),
        compressor=AverageLeafCompressor(hidden_size=1),
    )

    output = cache(
        hidden,
        query,
        token_ids=token_ids,
        query_token_ids=query_token_ids,
    )

    assert output.revived_leaf_indices.tolist() == [[[1]]]
    assert output.candidate_indices.tolist() == [[[4, 5, 6, 7, 8, 9]]]
    assert output.neural_leaf_scores is not None
    assert output.symbolic_leaf_scores is not None
    assert output.symbolic_leaf_scores.shape == output.leaf_scores.shape


def test_collect_probe_metrics_reports_score_channel_diagnostics() -> None:
    hidden = torch.zeros(1, 10, 1)
    query = torch.zeros(1, 1, 1)
    token_ids = torch.tensor([[65, 65, 65, 65, 88, 89, 90, 90, 10, 11]])
    query_token_ids = torch.tensor([[[88, 89, 90, 90]]])
    cache = RevivalMemoryCache(
        hidden_size=1,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=2,
            summary_slots=1,
            revived_top_k=1,
            symbolic_weight=5.0,
            symbolic_hash_bins=64,
            symbolic_ngrams=(2, 3),
        ),
        compressor=AverageLeafCompressor(hidden_size=1),
    )

    output = cache(
        hidden,
        query,
        token_ids=token_ids,
        query_token_ids=query_token_ids,
    )
    metrics = collect_probe_metrics(output, torch.tensor([[5]]), seq_len=10, leaf_size=4)

    assert metrics.combined_source_rank == 1.0
    assert metrics.symbolic_source_rank == 1.0
    assert metrics.symbolic_source_hit_rate == 1.0


def test_source_set_metrics_reward_multiple_recalled_leaves() -> None:
    hidden = torch.zeros(1, 16, 4)
    hidden[:, 0:4, 0] = 3.0
    hidden[:, 8:12, 0] = 2.5
    query = torch.zeros(1, 1, 4)
    query[:, :, 0] = 1.0
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=4,
            summary_slots=1,
            revived_top_k=2,
        ),
        compressor=AverageLeafCompressor(hidden_size=4),
    )

    output = cache(hidden, query)
    source_set = torch.tensor([[[1, 9]]])

    assert output.revived_leaf_indices.tolist() == [[[0, 2]]]
    assert memory_cache_source_set_leaf_recall(output, source_set, leaf_size=4).item() == 1.0
    assert memory_cache_source_set_leaf_accuracy(output, source_set, leaf_size=4).item() == 1.0
    assert memory_cache_source_set_leaf_loss(output, source_set, leaf_size=4).item() < 0.8
    assert torch.isfinite(
        memory_cache_source_set_survival_loss(
            output,
            source_set,
            leaf_size=4,
            top_k=2,
        )
    )
    metrics = collect_probe_set_metrics(output, source_set, seq_len=16, leaf_size=4)
    assert metrics.source_hit_rate == 1.0
    assert metrics.combined_source_rank is not None


def test_source_set_recall_is_fractional_when_only_one_leaf_survives() -> None:
    hidden = torch.zeros(1, 16, 4)
    hidden[:, 0:4, 0] = 3.0
    hidden[:, 8:12, 0] = 2.5
    query = torch.zeros(1, 1, 4)
    query[:, :, 0] = 1.0
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=4,
            summary_slots=1,
            revived_top_k=1,
        ),
        compressor=AverageLeafCompressor(hidden_size=4),
    )

    output = cache(hidden, query)
    source_set = torch.tensor([[[1, 9]]])

    assert memory_cache_source_set_leaf_recall(output, source_set, leaf_size=4).item() == 0.5


def test_source_set_coverage_loss_rewards_spread_across_targets() -> None:
    candidate_indices = torch.tensor([[[0, 1, 2, 3, 8, 9, 10, 11]]])
    candidate_mask = torch.ones_like(candidate_indices, dtype=torch.bool)
    base = dict(
        candidate_indices=candidate_indices,
        candidate_mask=candidate_mask,
        revived_leaf_indices=torch.tensor([[[0, 2]]]),
        revived_leaf_mask=torch.tensor([[[True, True]]]),
        leaf_summaries=torch.zeros(1, 3, 1, 2),
        leaf_token_indices=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
        leaf_token_mask=torch.ones(1, 3, 4, dtype=torch.bool),
        recent_start=12,
    )
    one_hot = RevivalCacheOutput(
        leaf_scores=torch.tensor([[[6.0, 0.0, 0.0]]]),
        **base,
    )
    covered = RevivalCacheOutput(
        leaf_scores=torch.tensor([[[4.0, 0.0, 4.0]]]),
        **base,
    )
    source_set = torch.tensor([[[1, 9]]])

    assert memory_cache_source_set_coverage_loss(
        covered,
        source_set,
        leaf_size=4,
    ) < memory_cache_source_set_coverage_loss(
        one_hot,
        source_set,
        leaf_size=4,
    )


def test_dynamic_top_k_keeps_fewer_leaves_when_scores_are_sharp() -> None:
    hidden = torch.tensor([[[5.0], [5.0], [4.6], [4.6], [0.0], [0.0], [9.0], [9.0]]])
    query = torch.ones(1, 1, 1)
    cache = RevivalMemoryCache(
        hidden_size=1,
        config=RevivalCacheConfig(
            leaf_size=2,
            recent_window=2,
            summary_slots=1,
            revived_top_k=3,
            revived_min_k=1,
            dynamic_score_margin=0.2,
        ),
        compressor=AverageLeafCompressor(hidden_size=1),
    )

    output = cache(hidden, query)

    assert output.revived_leaf_indices.tolist() == [[[0, 1, 2]]]
    assert output.revived_leaf_mask.tolist() == [[[True, False, False]]]
    assert output.active_exact_tokens_per_query.tolist() == [[4]]


def test_average_leaf_compressor_returns_one_slot() -> None:
    compressor = AverageLeafCompressor(hidden_size=3)
    leaves = torch.tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]]])
    mask = torch.tensor([[[True, True]]])

    summary = compressor(leaves, mask)

    assert summary.shape == (1, 1, 1, 3)
    assert summary.tolist() == [[[[2.0, 3.0, 4.0]]]]


def test_hybrid_leaf_compressor_uses_average_anchor_slot() -> None:
    compressor = HybridLeafCompressor(hidden_size=3, summary_slots=3)
    leaves = torch.tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]]])
    mask = torch.tensor([[[True, True]]])

    summary = compressor(leaves, mask)

    assert summary.shape == (1, 1, 3, 3)
    assert summary[:, :, 0].tolist() == [[[2.0, 3.0, 4.0]]]


def test_summary_diversity_loss_detects_collapsed_slots() -> None:
    hidden = torch.zeros(1, 12, 4)
    hidden[:, 0:8, 0] = 1.0
    query = torch.zeros(1, 1, 4)
    query[:, :, 0] = 1.0
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=4,
            summary_slots=2,
            revived_top_k=1,
        ),
    )
    with torch.no_grad():
        cache.compressor.slot_queries.zero_()
        cache.compressor.slot_queries[:, 0] = 1.0

    output = cache(hidden, query)

    assert memory_cache_summary_diversity_loss(output).item() > 0.9


def test_summary_diversity_loss_is_zero_for_single_slot() -> None:
    hidden = torch.randn(1, 12, 4)
    query = torch.randn(1, 1, 4)
    cache = RevivalMemoryCache(
        hidden_size=4,
        config=RevivalCacheConfig(
            leaf_size=4,
            recent_window=4,
            summary_slots=1,
            revived_top_k=1,
        ),
        compressor=AverageLeafCompressor(hidden_size=4),
    )

    output = cache(hidden, query)

    assert memory_cache_summary_diversity_loss(output).item() == 0.0


def test_memory_cache_route_probe_trains_one_source_hit_step() -> None:
    torch.manual_seed(123)
    input_ids = torch.randint(0, 32, (2, 64))
    answer_indices = torch.tensor([[60, 61], [60, 61]])
    source_indices = torch.tensor([[12, 13], [20, 21]])
    probe = MemoryCacheRouteProbe(
        vocab_size=64,
        hidden_size=16,
        max_seq_len=64,
        cache_config=RevivalCacheConfig(
            leaf_size=8,
            recent_window=16,
            summary_slots=2,
            revived_top_k=2,
        ),
        query_window=16,
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    output = probe(input_ids, answer_indices)
    loss = memory_cache_topk_survival_loss(
        output,
        source_indices,
        leaf_size=8,
        top_k=2,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert output.candidate_indices.shape[:2] == answer_indices.shape


def test_memory_cache_route_probe_accepts_hybrid_compressor() -> None:
    input_ids = torch.randint(0, 32, (2, 64))
    answer_indices = torch.tensor([[60, 61], [60, 61]])
    source_indices = torch.tensor([[12, 13], [20, 21]])
    probe = MemoryCacheRouteProbe(
        vocab_size=64,
        hidden_size=16,
        max_seq_len=64,
        cache_config=RevivalCacheConfig(
            leaf_size=8,
            recent_window=16,
            summary_slots=3,
            revived_top_k=2,
            slot_score_reduction="logsumexp",
            slot_score_temperature=0.7,
        ),
        query_window=16,
        compressor_type="hybrid",
    )

    output = probe(input_ids, answer_indices)
    loss = memory_cache_topk_survival_loss(
        output,
        source_indices,
        leaf_size=8,
        top_k=2,
    )

    assert torch.isfinite(loss)
    assert output.leaf_summaries.shape[2] == 3
    assert output.candidate_indices.shape[:2] == answer_indices.shape


@pytest.mark.parametrize("pooling", ["last", "attn", "mean-pos"])
def test_memory_cache_route_probe_query_pooling_modes(pooling: str) -> None:
    input_ids = torch.randint(0, 32, (2, 64))
    answer_indices = torch.tensor([[60, 61], [60, 61]])
    probe = MemoryCacheRouteProbe(
        vocab_size=64,
        hidden_size=16,
        max_seq_len=64,
        cache_config=RevivalCacheConfig(
            leaf_size=8,
            recent_window=16,
            summary_slots=1,
            revived_top_k=2,
        ),
        query_window=16,
        compressor_type="average",
        query_pooling=pooling,
        use_query_index_embedding=True,
    )

    output = probe(input_ids, answer_indices)

    assert output.candidate_indices.shape[:2] == answer_indices.shape


def test_usefulness_ledger_tracks_revived_leaves_with_decay() -> None:
    ledger = UsefulnessLedger(leaf_count=4, decay=0.5, revive_bonus=2.0)
    revived = torch.tensor([[[1, 3]], [[1, 0]]])
    mask = torch.tensor([[[True, False]], [[True, True]]])

    ledger.update(revived, mask)
    ledger.update(revived, mask)

    assert ledger.scores.tolist() == [3.0, 6.0, 0.0, 0.0]
    assert ledger.lowest_scoring_leaves(2).tolist() == [2, 3]


def test_scheduled_recent_window_linearly_tightens_cache() -> None:
    assert scheduled_recent_window(
        start_window=1024,
        end_window=256,
        step=1,
        curriculum_steps=4,
    ) == 1024
    assert scheduled_recent_window(
        start_window=1024,
        end_window=256,
        step=4,
        curriculum_steps=4,
    ) == 256
    assert scheduled_recent_window(
        start_window=1024,
        end_window=256,
        step=10,
        curriculum_steps=4,
    ) == 256
