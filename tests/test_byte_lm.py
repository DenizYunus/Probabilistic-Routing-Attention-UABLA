from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla.byte_lm import (
    BYTE_LM_VOCAB_SIZE,
    ByteLMConfig,
    ByteLanguageModelingDataset,
    ByteNeedleRecallConfig,
    ByteNeedleRecallDataset,
    RoutingBudgetStage,
    byte_answer_accuracy,
    byte_answer_cross_entropy,
    bytes_to_ids,
    ensure_min_bytes,
    evaluate_byte_lm,
    ids_to_bytes,
    make_byte_loader,
    split_byte_stream,
    train_byte_lm_steps,
)
from uabla.config import UABLAConfig
from uabla.model import TinyTransformerLM


def test_byte_ids_round_trip_raw_bytes() -> None:
    data = bytes([0, 1, 2, 127, 128, 255])
    ids = bytes_to_ids(data)

    assert ids.tolist() == [0, 1, 2, 127, 128, 255]
    assert ids_to_bytes(ids) == data


def test_byte_dataset_returns_shifted_next_byte_labels() -> None:
    config = ByteLMConfig(seq_len=3, dataset_size=1, seed=123)
    dataset = ByteLanguageModelingDataset(bytes_to_ids(b"abcd"), config)

    sample = dataset[0]

    assert ids_to_bytes(sample["input_ids"]) == b"abc"
    assert ids_to_bytes(sample["labels"]) == b"bcd"


def test_byte_needle_dataset_places_answer_after_gap() -> None:
    config = ByteNeedleRecallConfig(
        seq_len=96,
        dataset_size=2,
        seed=123,
        code_length=4,
        min_gap=24,
    )
    dataset = ByteNeedleRecallDataset(bytes_to_ids(ensure_min_bytes(b"story text ", 256)), config)

    sample = dataset[0]

    assert sample["input_ids"].shape == (config.seq_len,)
    assert sample["labels"].shape == (config.seq_len,)
    assert sample["answer_index"].shape == (config.code_length,)
    assert sample["answer_token"].shape == (config.code_length,)
    assert sample["answer_source_index"].shape == (config.code_length,)
    assert int(sample["needle_gap"].item()) >= config.min_gap
    for answer_idx, answer_token in zip(
        sample["answer_index"].tolist(),
        sample["answer_token"].tolist(),
        strict=True,
    ):
        assert int(sample["labels"][answer_idx].item()) == answer_token


def test_byte_answer_metrics_use_answer_positions() -> None:
    config = ByteNeedleRecallConfig(
        seq_len=96,
        dataset_size=2,
        seed=123,
        code_length=4,
        min_gap=24,
    )
    dataset = ByteNeedleRecallDataset(bytes_to_ids(ensure_min_bytes(b"story text ", 256)), config)
    batch = {
        key: torch.stack([dataset[idx][key] for idx in range(2)])
        for key in [
            "input_ids",
            "labels",
            "answer_index",
            "answer_token",
            "answer_source_index",
            "needle_gap",
            "answer_random_chance",
        ]
    }
    logits = torch.zeros(2, config.seq_len, BYTE_LM_VOCAB_SIZE)
    batch_indices = torch.arange(2).view(2, 1).expand_as(batch["answer_index"])
    logits[batch_indices, batch["answer_index"], batch["answer_token"]] = 10.0

    loss = byte_answer_cross_entropy(logits, batch)
    accuracy = byte_answer_accuracy(logits, batch)

    assert loss.item() < 0.02
    assert accuracy.item() == 1.0


def test_byte_stream_helpers_repeat_and_split() -> None:
    data = ensure_min_bytes(b"abc", min_length=12)
    train_ids, eval_ids = split_byte_stream(bytes_to_ids(data), seq_len=4, eval_fraction=0.25)

    assert len(data) >= 12
    assert train_ids.numel() >= 5
    assert eval_ids.numel() >= 5


def test_train_and_evaluate_one_tiny_byte_lm_step() -> None:
    torch.manual_seed(8)
    byte_ids = bytes_to_ids(ensure_min_bytes(b"byte uabla test ", min_length=128))
    train_ids, eval_ids = split_byte_stream(byte_ids, seq_len=16, eval_fraction=0.25)
    loader = make_byte_loader(
        ByteLanguageModelingDataset(
            train_ids,
            ByteLMConfig(seq_len=16, dataset_size=8, seed=8),
        ),
        batch_size=2,
        shuffle=False,
    )
    eval_loader = make_byte_loader(
        ByteLanguageModelingDataset(
            eval_ids,
            ByteLMConfig(seq_len=16, dataset_size=4, seed=18),
        ),
        batch_size=2,
        shuffle=False,
    )
    model = TinyTransformerLM(
        vocab_size=BYTE_LM_VOCAB_SIZE,
        max_seq_len=16,
        hidden_size=16,
        num_layers=1,
        attention_type="dense",
        input_mixer_kernel=3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = train_byte_lm_steps(
        model,
        loader,
        steps=1,
        optimizer=optimizer,
        device=torch.device("cpu"),
        log_every=1,
    )
    metrics = evaluate_byte_lm(model, eval_loader, device=torch.device("cpu"), batches=1)

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0].loss))
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert 0.0 <= metrics.byte_accuracy <= 1.0
    assert metrics.byte_perplexity > 0
    assert metrics.cache_dim_per_token_per_layer == 2 * model.hidden_size


def test_train_and_evaluate_one_tiny_needle_step() -> None:
    torch.manual_seed(9)
    byte_ids = bytes_to_ids(ensure_min_bytes(b"needle story text ", min_length=256))
    train_ids, eval_ids = split_byte_stream(byte_ids, seq_len=96, eval_fraction=0.25)
    loader = make_byte_loader(
        ByteNeedleRecallDataset(
            train_ids,
            ByteNeedleRecallConfig(seq_len=96, dataset_size=4, seed=9, code_length=4, min_gap=24),
        ),
        batch_size=2,
        shuffle=False,
    )
    eval_loader = make_byte_loader(
        ByteNeedleRecallDataset(
            eval_ids,
            ByteNeedleRecallConfig(seq_len=96, dataset_size=4, seed=19, code_length=4, min_gap=24),
        ),
        batch_size=2,
        shuffle=False,
    )
    model = TinyTransformerLM(
        vocab_size=BYTE_LM_VOCAB_SIZE,
        max_seq_len=96,
        hidden_size=16,
        num_layers=1,
        attention_type="dense",
        input_mixer_kernel=3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = train_byte_lm_steps(
        model,
        loader,
        steps=1,
        optimizer=optimizer,
        device=torch.device("cpu"),
        lm_loss_weight=0.2,
        answer_loss_weight=1.0,
        log_every=1,
    )
    metrics = evaluate_byte_lm(model, eval_loader, device=torch.device("cpu"), batches=1)

    assert len(history) == 1
    assert history[0].answer_loss is not None
    assert metrics.answer_accuracy is not None
    assert 0.0 <= metrics.answer_accuracy <= 1.0


def test_stage_training_enables_shifted_routing() -> None:
    torch.manual_seed(10)
    byte_ids = bytes_to_ids(ensure_min_bytes(b"stage shifted byte test ", min_length=128))
    train_ids, _ = split_byte_stream(byte_ids, seq_len=16, eval_fraction=0.25)
    loader = make_byte_loader(
        ByteLanguageModelingDataset(
            train_ids,
            ByteLMConfig(seq_len=16, dataset_size=4, seed=10),
        ),
        batch_size=2,
        shuffle=False,
    )
    config = UABLAConfig(
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
        use_shifted_blocks=True,
    )
    model = TinyTransformerLM(
        vocab_size=BYTE_LM_VOCAB_SIZE,
        max_seq_len=16,
        hidden_size=16,
        num_layers=1,
        attention_type="uabla",
        local_window=4,
        uabla_config=config,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    events: list[dict[str, object]] = []

    train_byte_lm_steps(
        model,
        loader,
        steps=2,
        optimizer=optimizer,
        device=torch.device("cpu"),
        log_every=2,
        shifted_enable_step=1,
        stage_callback=events.append,
    )

    assert events == [
        {
            "event": "enable_shifted_routing",
            "reason": "fixed_step",
            "shifted_routing_blocks": True,
            "step": 2,
        }
    ]
    assert model.uabla_config is not None
    assert model.uabla_config.use_shifted_blocks


def test_stage_training_can_wait_for_answer_accuracy_threshold() -> None:
    torch.manual_seed(11)
    byte_ids = bytes_to_ids(ensure_min_bytes(b"adaptive shifted byte needle test ", min_length=256))
    train_ids, _ = split_byte_stream(byte_ids, seq_len=96, eval_fraction=0.25)
    loader = make_byte_loader(
        ByteNeedleRecallDataset(
            train_ids,
            ByteNeedleRecallConfig(seq_len=96, dataset_size=4, seed=11, code_length=4, min_gap=24),
        ),
        batch_size=2,
        shuffle=False,
    )
    config = UABLAConfig(
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
        use_shifted_blocks=True,
    )
    model = TinyTransformerLM(
        vocab_size=BYTE_LM_VOCAB_SIZE,
        max_seq_len=96,
        hidden_size=16,
        num_layers=1,
        attention_type="uabla",
        local_window=4,
        uabla_config=config,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    events: list[dict[str, object]] = []

    train_byte_lm_steps(
        model,
        loader,
        steps=2,
        optimizer=optimizer,
        device=torch.device("cpu"),
        lm_loss_weight=0.2,
        answer_loss_weight=1.0,
        log_every=1,
        shifted_enable_answer_accuracy=0.0,
        stage_callback=events.append,
    )

    assert events
    assert events[0]["reason"] == "answer_accuracy_threshold"
    assert events[0]["step"] == 2
    assert events[0]["trigger_step"] == 1
    assert model.uabla_config is not None
    assert model.uabla_config.use_shifted_blocks


def test_route_budget_curriculum_updates_buckets_during_training() -> None:
    torch.manual_seed(12)
    byte_ids = bytes_to_ids(ensure_min_bytes(b"budget curriculum byte test ", min_length=128))
    train_ids, _ = split_byte_stream(byte_ids, seq_len=16, eval_fraction=0.25)
    loader = make_byte_loader(
        ByteLanguageModelingDataset(
            train_ids,
            ByteLMConfig(seq_len=16, dataset_size=4, seed=12),
        ),
        batch_size=2,
        shuffle=False,
    )
    config = UABLAConfig(
        hidden_size=16,
        routing_dim=8,
        value_dim=12,
        position_dim=4,
        block_size=4,
        local_window=4,
        centroid_hit_buckets=(4,),
        token_k_buckets=(4,),
        superblock_size_blocks=2,
        superblock_hit_buckets=(2,),
        use_shifted_blocks=False,
    )
    model = TinyTransformerLM(
        vocab_size=BYTE_LM_VOCAB_SIZE,
        max_seq_len=16,
        hidden_size=16,
        num_layers=1,
        attention_type="uabla",
        local_window=4,
        uabla_config=config,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    events: list[dict[str, object]] = []

    train_byte_lm_steps(
        model,
        loader,
        steps=2,
        optimizer=optimizer,
        device=torch.device("cpu"),
        log_every=2,
        route_budget_stages=(
            RoutingBudgetStage(
                start_step=1,
                superblock_hit_buckets=(2,),
                centroid_hit_buckets=(4,),
                token_k_buckets=(4,),
            ),
            RoutingBudgetStage(
                start_step=2,
                superblock_hit_buckets=(2, 4),
                centroid_hit_buckets=(4, 8),
                token_k_buckets=(4, 8),
            ),
        ),
        route_entropy_weight=0.01,
        route_entropy_min=0.25,
        stage_callback=events.append,
    )

    assert [event["event"] for event in events] == [
        "set_route_budget_stage",
        "set_route_budget_stage",
    ]
    assert model.uabla_config is not None
    assert model.uabla_config.centroid_hit_buckets == (4, 8)
    assert model.uabla_config.token_k_buckets == (4, 8)
