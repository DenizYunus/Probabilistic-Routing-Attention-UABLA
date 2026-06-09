from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla.synthetic import (
    KeyValueRecallConfig,
    KeyValueRecallDataset,
    answer_accuracy,
    answer_cross_entropy,
    value_answer_accuracy,
    value_answer_cross_entropy,
    value_random_chance,
)


def test_key_value_recall_sample_has_final_answer_target() -> None:
    config = KeyValueRecallConfig(seq_len=32, num_pairs=4, dataset_size=4, seed=123)
    dataset = KeyValueRecallDataset(config)

    sample = dataset[0]

    assert sample["input_ids"].shape == (config.model_seq_len,)
    assert sample["labels"].shape == (config.model_seq_len,)
    assert int(sample["input_ids"][-2].item()) == config.query_token
    assert int(sample["input_ids"][-1].item()) == int(sample["query_key"][0].item())
    assert int(sample["labels"][-1].item()) == int(sample["answer_token"][0].item())
    assert sample["answer_index"].shape == (1,)
    assert sample["answer_source_index"].shape == (1,)
    assert int(sample["answer_index"][0].item()) == config.model_seq_len - 1
    assert int(sample["answer_source_index"][0].item()) % 3 == 1
    assert int(sample["value_start"].item()) == config.value_start
    assert int(sample["num_values"].item()) == config.num_values


def test_key_value_recall_sample_can_have_multiple_queries() -> None:
    config = KeyValueRecallConfig(seq_len=48, num_pairs=5, num_queries=3, dataset_size=4, seed=123)
    dataset = KeyValueRecallDataset(config)

    sample = dataset[0]

    assert sample["answer_index"].shape == (config.num_queries,)
    assert sample["answer_token"].shape == (config.num_queries,)
    assert sample["answer_source_index"].shape == (config.num_queries,)
    assert sample["query_key"].shape == (config.num_queries,)
    for query_idx in range(config.num_queries):
        answer_index = int(sample["answer_index"][query_idx].item())
        source_index = int(sample["answer_source_index"][query_idx].item())
        assert int(sample["input_ids"][answer_index - 1].item()) == config.query_token
        assert int(sample["input_ids"][answer_index].item()) == int(
            sample["query_key"][query_idx].item(),
        )
        assert int(sample["labels"][answer_index].item()) == int(
            sample["answer_token"][query_idx].item(),
        )
        assert int(sample["input_ids"][source_index].item()) == int(
            sample["answer_token"][query_idx].item(),
        )


def test_answer_metrics_use_answer_position() -> None:
    config = KeyValueRecallConfig(seq_len=36, num_pairs=4, num_queries=3, dataset_size=2)
    batch = {
        key: torch.stack([KeyValueRecallDataset(config)[idx][key] for idx in range(2)])
        for key in [
            "input_ids",
            "labels",
            "answer_index",
            "answer_token",
            "answer_source_index",
            "query_key",
            "value_start",
            "num_values",
        ]
    }
    logits = torch.zeros(2, config.model_seq_len, config.vocab_size)
    batch_indices = torch.arange(2).view(2, 1).expand_as(batch["answer_index"])
    logits[batch_indices, batch["answer_index"], batch["answer_token"]] = 10.0

    loss = answer_cross_entropy(logits, batch)
    accuracy = answer_accuracy(logits, batch)
    value_loss = value_answer_cross_entropy(logits, batch)
    value_accuracy = value_answer_accuracy(logits, batch)

    assert loss.item() < 0.01
    assert accuracy.item() == 1.0
    assert value_loss.item() < 0.01
    assert value_accuracy.item() == 1.0
    assert value_random_chance(batch) == 1.0 / config.num_values
