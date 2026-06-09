"""Synthetic long-memory tasks for testing UABLA routing."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - lets non-torch config tests import the package.
    torch = None
    Dataset = object


@dataclass(frozen=True)
class KeyValueRecallConfig:
    """Configuration for fixed-length key-value recall samples."""

    seq_len: int = 128
    num_pairs: int = 8
    num_queries: int = 1
    num_keys: int = 64
    num_values: int = 64
    filler_vocab: int = 32
    dataset_size: int = 2048
    seed: int = 0

    def __post_init__(self) -> None:
        if self.seq_len < self.num_pairs * 3 + self.num_queries * 3:
            raise ValueError("seq_len must fit all key/value pairs plus query triples")
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if self.num_pairs > self.num_keys:
            raise ValueError("num_pairs cannot exceed num_keys")
        if self.num_keys <= 0 or self.num_values <= 0 or self.filler_vocab <= 0:
            raise ValueError("vocab sizes must be positive")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")

    @property
    def filler_start(self) -> int:
        return 1

    @property
    def key_start(self) -> int:
        return self.filler_start + self.filler_vocab

    @property
    def value_start(self) -> int:
        return self.key_start + self.num_keys

    @property
    def sep_token(self) -> int:
        return self.value_start + self.num_values

    @property
    def query_token(self) -> int:
        return self.sep_token + 1

    @property
    def vocab_size(self) -> int:
        return self.query_token + 1

    @property
    def model_seq_len(self) -> int:
        return self.seq_len - 1


class KeyValueRecallDataset(Dataset):
    """Generate sequences where query keys ask for earlier values.

    Full sequence format:

    ```text
    key value sep key value sep ... filler ... QUERY key value ...
    ```

    Models receive `tokens[:-1]` and must predict `tokens[1:]`. Answer accuracy
    is measured at each query-key input position, where the target is the value.
    """

    def __init__(self, config: KeyValueRecallConfig) -> None:
        if torch is None:
            raise ModuleNotFoundError("KeyValueRecallDataset requires torch")
        self.config = config

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        config = self.config
        generator = torch.Generator()
        generator.manual_seed(config.seed + index)

        tokens = torch.randint(
            low=config.filler_start,
            high=config.filler_start + config.filler_vocab,
            size=(config.seq_len,),
            generator=generator,
            dtype=torch.long,
        )

        keys = torch.randperm(config.num_keys, generator=generator)[: config.num_pairs]
        values = torch.randint(
            low=0,
            high=config.num_values,
            size=(config.num_pairs,),
            generator=generator,
            dtype=torch.long,
        )
        pair_tokens = torch.empty(config.num_pairs * 3, dtype=torch.long)
        pair_tokens[0::3] = config.key_start + keys
        pair_tokens[1::3] = config.value_start + values
        pair_tokens[2::3] = config.sep_token
        tokens[: pair_tokens.numel()] = pair_tokens

        if config.num_queries <= config.num_pairs:
            query_pair_indices = torch.randperm(config.num_pairs, generator=generator)[
                : config.num_queries
            ]
        else:
            query_pair_indices = torch.randint(
                config.num_pairs,
                (config.num_queries,),
                generator=generator,
                dtype=torch.long,
            )
        query_keys = config.key_start + keys[query_pair_indices]
        answer_tokens = config.value_start + values[query_pair_indices]
        answer_source_indices = query_pair_indices * 3 + 1
        query_start = config.seq_len - config.num_queries * 3
        query_offsets = torch.arange(config.num_queries, dtype=torch.long) * 3
        tokens[query_start + query_offsets] = config.query_token
        tokens[query_start + query_offsets + 1] = query_keys
        tokens[query_start + query_offsets + 2] = answer_tokens
        answer_indices = query_start + query_offsets + 1

        return {
            "input_ids": tokens[:-1].clone(),
            "labels": tokens[1:].clone(),
            "answer_index": answer_indices.clone(),
            "answer_token": answer_tokens.clone(),
            "answer_source_index": answer_source_indices.clone(),
            "query_key": query_keys.clone(),
            "value_start": torch.tensor(config.value_start, dtype=torch.long),
            "num_values": torch.tensor(config.num_values, dtype=torch.long),
        }


def answer_cross_entropy(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Cross-entropy at the synthetic task's answer positions."""

    if torch is None:
        raise ModuleNotFoundError("answer_cross_entropy requires torch")
    answer_logits, answer_targets = _gather_answer_logits(logits, batch)
    return torch.nn.functional.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_targets.reshape(-1),
    )


def answer_accuracy(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Accuracy at the synthetic task's answer positions."""

    if torch is None:
        raise ModuleNotFoundError("answer_accuracy requires torch")
    answer_logits, answer_targets = _gather_answer_logits(logits, batch)
    predictions = answer_logits.argmax(dim=-1)
    return (predictions == answer_targets).float().mean()


def value_answer_cross_entropy(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Cross-entropy at answer positions restricted to valid value tokens."""

    if torch is None:
        raise ModuleNotFoundError("value_answer_cross_entropy requires torch")
    value_start = int(batch["value_start"][0].item())
    num_values = int(batch["num_values"][0].item())
    answer_logits, answer_targets = _gather_answer_logits(logits, batch)
    answer_logits = answer_logits[..., value_start : value_start + num_values]
    value_targets = answer_targets - value_start
    return torch.nn.functional.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        value_targets.reshape(-1),
    )


def value_answer_accuracy(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Accuracy at answer positions after restricting predictions to value tokens."""

    if torch is None:
        raise ModuleNotFoundError("value_answer_accuracy requires torch")
    value_start = int(batch["value_start"][0].item())
    num_values = int(batch["num_values"][0].item())
    answer_logits, answer_targets = _gather_answer_logits(logits, batch)
    answer_logits = answer_logits[..., value_start : value_start + num_values]
    predictions = answer_logits.argmax(dim=-1) + value_start
    return (predictions == answer_targets).float().mean()


def value_random_chance(batch: dict[str, torch.Tensor]) -> float:
    """Random-answer baseline for value-only prediction."""

    return 1.0 / float(int(batch["num_values"][0].item()))


def _gather_answer_logits(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    answer_indices = batch["answer_index"].to(logits.device)
    answer_targets = batch["answer_token"].to(logits.device)
    if answer_indices.ndim == 1:
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        return logits[batch_indices, answer_indices], answer_targets
    if answer_indices.ndim == 2:
        batch_indices = torch.arange(logits.shape[0], device=logits.device).view(-1, 1)
        batch_indices = batch_indices.expand_as(answer_indices)
        return logits[batch_indices, answer_indices], answer_targets
    raise ValueError("answer_index must have shape [batch] or [batch, queries]")
