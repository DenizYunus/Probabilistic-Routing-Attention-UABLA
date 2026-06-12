from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla.byte_lm import (
    BYTE_LM_VOCAB_SIZE,
    ByteLMConfig,
    ByteLanguageModelingDataset,
    bytes_to_ids,
    ensure_min_bytes,
    evaluate_byte_lm,
    ids_to_bytes,
    make_byte_loader,
    split_byte_stream,
    train_byte_lm_steps,
)
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
