from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from uabla.config import UABLAConfig
from uabla.model import TinyTransformerLM


def test_tiny_model_forward_for_all_attention_types() -> None:
    vocab_size = 32
    seq_len = 15
    input_ids = torch.randint(0, vocab_size, (2, seq_len))

    for attention_type in ["dense", "local", "uabla"]:
        uabla_config = None
        if attention_type == "uabla":
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
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            hidden_size=16,
            num_layers=1,
            attention_type=attention_type,
            local_window=4,
            uabla_config=uabla_config,
        )
        output = model(input_ids, return_routing=True, return_attention=True)

        assert output.logits.shape == (2, seq_len, vocab_size)
        if attention_type == "uabla":
            assert output.uabla_outputs
            assert model.cache_dim_per_token_per_layer == uabla_config.cache_dim
        else:
            assert output.dense_attentions
            assert model.cache_dim_per_token_per_layer == 2 * model.hidden_size


def test_token_scores_are_returned_only_for_final_uabla_layer() -> None:
    vocab_size = 32
    seq_len = 15
    input_ids = torch.randint(0, vocab_size, (2, seq_len))
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
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        hidden_size=16,
        num_layers=2,
        attention_type="uabla",
        local_window=4,
        uabla_config=uabla_config,
    )

    output = model(input_ids, return_routing=True, return_token_scores=True)

    assert len(output.uabla_outputs) == 2
    assert output.uabla_outputs[0].token_scores is None
    assert output.uabla_outputs[1].token_scores is not None
