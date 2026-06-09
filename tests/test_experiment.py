from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla.config import UABLAConfig
from uabla.experiment import evaluate, make_loader, train_steps
from uabla.model import TinyTransformerLM
from uabla.synthetic import KeyValueRecallConfig, KeyValueRecallDataset


def test_train_and_evaluate_one_tiny_uabla_step() -> None:
    torch.manual_seed(5)
    data_config = KeyValueRecallConfig(seq_len=32, num_pairs=4, dataset_size=8, seed=5)
    loader = make_loader(KeyValueRecallDataset(data_config), batch_size=2, shuffle=False)
    uabla_config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=4,
        centroid_hit_buckets=(4, 8),
        token_k_buckets=(2, 4),
        superblock_size_blocks=2,
        superblock_hit_buckets=(2, 4),
    )
    model = TinyTransformerLM(
        vocab_size=data_config.vocab_size,
        max_seq_len=data_config.model_seq_len,
        hidden_size=16,
        num_layers=1,
        attention_type="uabla",
        local_window=4,
        uabla_config=uabla_config,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = train_steps(
        model,
        loader,
        steps=1,
        optimizer=optimizer,
        device=torch.device("cpu"),
        log_every=1,
        direct_route_weight=0.1,
        budget_weight=0.01,
    )
    metrics = evaluate(model, loader, device=torch.device("cpu"), batches=1)

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0].loss))
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert 0.0 <= metrics.answer_accuracy <= 1.0
    assert 0.0 <= metrics.value_answer_accuracy <= 1.0
    assert metrics.value_random_chance == 1.0 / data_config.num_values
    assert metrics.cache_dim_per_token_per_layer == uabla_config.cache_dim
