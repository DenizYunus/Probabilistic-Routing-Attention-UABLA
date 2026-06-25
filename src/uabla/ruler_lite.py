"""RULER-inspired byte-level retrieval datasets for cache experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Dataset = object

from .byte_lm import NEEDLE_ALPHABET, bytes_to_ids

RULER_LITE_TASKS = ("multi_needle", "multi_hop", "aggregation")
RULER_LITE_TASK_TO_CLI = {
    "multi_needle": "ruler-multineedle",
    "multi_hop": "ruler-multihop",
    "aggregation": "ruler-aggregation",
}
RULER_LITE_CLI_TO_TASK = {value: key for key, value in RULER_LITE_TASK_TO_CLI.items()}


@dataclass(frozen=True)
class ByteRulerLiteConfig:
    """Configuration for RULER-inspired byte-level retrieval stress tests."""

    seq_len: int = 4096
    dataset_size: int = 4096
    seed: int = 0
    task: str = "multi_needle"
    code_length: int = 12
    key_length: int = 6
    num_needles: int = 8
    aggregation_count: int = 3
    min_gap: int = 512

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if self.task not in RULER_LITE_TASKS:
            raise ValueError(f"task must be one of {RULER_LITE_TASKS}")
        if self.code_length <= 0:
            raise ValueError("code_length must be positive")
        if self.key_length <= 0:
            raise ValueError("key_length must be positive")
        if self.num_needles <= 0:
            raise ValueError("num_needles must be positive")
        if self.aggregation_count <= 0:
            raise ValueError("aggregation_count must be positive")
        if self.task == "aggregation" and self.num_needles < self.aggregation_count:
            raise ValueError("aggregation task requires num_needles >= aggregation_count")
        if self.min_gap <= 0:
            raise ValueError("min_gap must be positive")
        source_len, answer_len = _ruler_lite_min_lengths(self)
        if self.seq_len + 1 < source_len + self.min_gap + answer_len:
            raise ValueError("seq_len is too short for RULER-lite source, gap, and answer")


class ByteRulerLiteDataset(Dataset):
    """RULER-inspired byte tasks beyond single needle retrieval."""

    def __init__(self, byte_ids: torch.Tensor, config: ByteRulerLiteConfig) -> None:
        if torch is None:
            raise ModuleNotFoundError("ByteRulerLiteDataset requires torch")
        if byte_ids.ndim != 1:
            raise ValueError("byte_ids must be a 1D tensor")
        if byte_ids.numel() == 0:
            raise ValueError("byte stream cannot be empty")
        if int(byte_ids.min().item()) < 0 or int(byte_ids.max().item()) > 255:
            raise ValueError("raw byte IDs must be in [0, 255]")
        self.byte_ids = byte_ids.to(dtype=torch.long).cpu()
        self.config = config

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        config = self.config
        generator = torch.Generator()
        generator.manual_seed(config.seed + index)
        full_len = config.seq_len + 1
        tokens = self._sample_background(full_len, generator)
        source, prompt, answer, source_offsets = self._build_ruler_example(generator)

        answer_segment = torch.cat([bytes_to_ids(prompt), answer])
        answer_segment_start = full_len - answer_segment.numel()
        latest_source_start = answer_segment_start - config.min_gap - source.numel()
        if latest_source_start < 0:
            raise RuntimeError("invalid RULER-lite layout; config validation should have caught this")
        if latest_source_start == 0:
            source_start = 0
        else:
            source_start = int(
                torch.randint(
                    latest_source_start + 1,
                    (1,),
                    generator=generator,
                    dtype=torch.long,
                ).item(),
            )

        tokens[source_start : source_start + source.numel()] = source
        tokens[answer_segment_start:] = answer_segment

        answer_start = answer_segment_start + len(prompt)
        answer_positions = torch.arange(
            answer_start,
            answer_start + answer.numel(),
            dtype=torch.long,
        )
        source_positions = source_start + source_offsets

        return {
            "input_ids": tokens[:-1].clone(),
            "labels": tokens[1:].clone(),
            "answer_index": answer_positions - 1,
            "answer_token": answer.clone(),
            "answer_source_index": source_positions.clone(),
            "needle_gap": torch.tensor(answer_start - int(source_positions[0].item()), dtype=torch.long),
            "answer_random_chance": torch.tensor(
                1.0 / float(len(NEEDLE_ALPHABET)),
                dtype=torch.float32,
            ),
            "ruler_task_id": torch.tensor(RULER_LITE_TASKS.index(config.task), dtype=torch.long),
            "ruler_num_needles": torch.tensor(config.num_needles, dtype=torch.long),
        }

    def _sample_background(self, length: int, generator: torch.Generator) -> torch.Tensor:
        if self.byte_ids.numel() >= length:
            max_start = self.byte_ids.numel() - length
            start = int(
                torch.randint(max_start + 1, (1,), generator=generator, dtype=torch.long).item(),
            )
            return self.byte_ids[start : start + length].clone()
        repeats = math.ceil(length / self.byte_ids.numel())
        return self.byte_ids.repeat(repeats)[:length].clone()

    def _build_ruler_example(
        self,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, bytes, torch.Tensor, torch.Tensor]:
        if self.config.task == "multi_needle":
            return self._build_multi_needle(generator)
        if self.config.task == "multi_hop":
            return self._build_multi_hop(generator)
        if self.config.task == "aggregation":
            return self._build_aggregation(generator)
        raise ValueError(f"unknown RULER-lite task: {self.config.task}")

    def _build_multi_needle(
        self,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, bytes, torch.Tensor, torch.Tensor]:
        config = self.config
        keys = _sample_unique_codes(
            generator,
            count=config.num_needles,
            length=config.key_length,
        )
        values = _sample_unique_codes(
            generator,
            count=config.num_needles,
            length=config.code_length,
        )
        target_idx = _randint(generator, config.num_needles)
        order = torch.randperm(config.num_needles, generator=generator).tolist()
        pieces: list[bytes] = []
        source_offsets: list[int] | None = None
        cursor = 0
        for idx in order:
            prefix = b"\nKEY " + keys[idx] + b" VALUE "
            record = prefix + values[idx]
            if idx == target_idx:
                source_offsets = list(range(cursor + len(prefix), cursor + len(record)))
            pieces.append(record)
            cursor += len(record)
        if source_offsets is None:
            raise RuntimeError("target record was not generated")
        prompt = b"\nQUESTION: value for KEY " + keys[target_idx] + b"?\nANSWER: "
        return (
            bytes_to_ids(b"".join(pieces)),
            prompt,
            bytes_to_ids(values[target_idx]),
            torch.tensor(source_offsets, dtype=torch.long),
        )

    def _build_multi_hop(
        self,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, bytes, torch.Tensor, torch.Tensor]:
        config = self.config
        start_keys = _sample_unique_codes(
            generator,
            count=config.num_needles,
            length=config.key_length,
        )
        middle_keys = _sample_unique_codes(
            generator,
            count=config.num_needles,
            length=config.key_length,
        )
        values = _sample_unique_codes(
            generator,
            count=config.num_needles,
            length=config.code_length,
        )
        target_idx = _randint(generator, config.num_needles)
        records: list[tuple[int, bytes, bool]] = []
        for idx in range(config.num_needles):
            records.append((idx, b"\nLINK " + start_keys[idx] + b" -> " + middle_keys[idx], False))
            records.append((idx, b"\nVALUE " + middle_keys[idx] + b" = " + values[idx], True))
        order = torch.randperm(len(records), generator=generator).tolist()
        pieces: list[bytes] = []
        source_offsets: list[int] | None = None
        cursor = 0
        for order_idx in order:
            idx, record, is_value = records[order_idx]
            if idx == target_idx and is_value:
                value_start = len(record) - config.code_length
                source_offsets = list(range(cursor + value_start, cursor + len(record)))
            pieces.append(record)
            cursor += len(record)
        if source_offsets is None:
            raise RuntimeError("target value record was not generated")
        prompt = b"\nQUESTION: follow KEY " + start_keys[target_idx] + b".\nANSWER: "
        return (
            bytes_to_ids(b"".join(pieces)),
            prompt,
            bytes_to_ids(values[target_idx]),
            torch.tensor(source_offsets, dtype=torch.long),
        )

    def _build_aggregation(
        self,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, bytes, torch.Tensor, torch.Tensor]:
        config = self.config
        group_count = max(2, config.num_needles - config.aggregation_count + 1)
        group_keys = _sample_unique_codes(
            generator,
            count=group_count,
            length=config.key_length,
        )
        target_key = group_keys[0]
        target_values = _sample_unique_codes(
            generator,
            count=config.aggregation_count,
            length=config.code_length,
        )
        distractor_count = config.num_needles - config.aggregation_count
        distractor_values = _sample_unique_codes(
            generator,
            count=max(distractor_count, 1),
            length=config.code_length,
        )
        records: list[tuple[int | None, bytes]] = []
        for part_idx, value in enumerate(target_values):
            records.append(
                (
                    part_idx,
                    _aggregation_record(target_key, part_idx, value),
                )
            )
        for idx in range(distractor_count):
            group_key = group_keys[1 + (idx % (group_count - 1))]
            records.append(
                (
                    None,
                    _aggregation_record(group_key, idx, distractor_values[idx]),
                )
            )
        order = torch.randperm(len(records), generator=generator).tolist()
        pieces: list[bytes] = []
        source_offsets_by_part: dict[int, list[int]] = {}
        cursor = 0
        for order_idx in order:
            part_idx, record = records[order_idx]
            if part_idx is not None:
                value_start = len(record) - config.code_length
                source_offsets_by_part[part_idx] = list(
                    range(cursor + value_start, cursor + len(record))
                )
            pieces.append(record)
            cursor += len(record)
        source_offsets: list[int] = []
        for part_idx in range(config.aggregation_count):
            source_offsets.extend(source_offsets_by_part[part_idx])
        prompt = b"\nQUESTION: concatenate GROUP " + target_key + b" in part order.\nANSWER: "
        answer = b"".join(target_values)
        return (
            bytes_to_ids(b"".join(pieces)),
            prompt,
            bytes_to_ids(answer),
            torch.tensor(source_offsets, dtype=torch.long),
        )


def _randint(generator: torch.Generator, high: int) -> int:
    return int(torch.randint(high, (1,), generator=generator, dtype=torch.long).item())


def _sample_unique_codes(
    generator: torch.Generator,
    *,
    count: int,
    length: int,
) -> list[bytes]:
    alphabet = torch.tensor(list(NEEDLE_ALPHABET), dtype=torch.long)
    values: list[bytes] = []
    seen: set[bytes] = set()
    max_unique = len(NEEDLE_ALPHABET) ** length
    if count > max_unique:
        raise ValueError("requested more unique codes than the alphabet can represent")
    while len(values) < count:
        indices = torch.randint(
            len(NEEDLE_ALPHABET),
            (length,),
            generator=generator,
            dtype=torch.long,
        )
        candidate = bytes(int(value) for value in alphabet[indices].tolist())
        if candidate in seen:
            continue
        seen.add(candidate)
        values.append(candidate)
    return values


def _aggregation_record(group_key: bytes, part_idx: int, value: bytes) -> bytes:
    part = f"{part_idx:02d}".encode("ascii")
    return b"\nGROUP " + group_key + b" PART " + part + b" = " + value


def _ruler_lite_min_lengths(config: ByteRulerLiteConfig) -> tuple[int, int]:
    key = b"K" * config.key_length
    code = b"C" * config.code_length
    if config.task == "multi_needle":
        record = b"\nKEY " + key + b" VALUE " + code
        prompt = b"\nQUESTION: value for KEY " + key + b"?\nANSWER: "
        return len(record) * config.num_needles, len(prompt) + config.code_length
    if config.task == "multi_hop":
        link = b"\nLINK " + key + b" -> " + key
        value = b"\nVALUE " + key + b" = " + code
        prompt = b"\nQUESTION: follow KEY " + key + b".\nANSWER: "
        return (len(link) + len(value)) * config.num_needles, len(prompt) + config.code_length
    if config.task == "aggregation":
        record = _aggregation_record(key, 0, code)
        prompt = b"\nQUESTION: concatenate GROUP " + key + b" in part order.\nANSWER: "
        answer_len = config.code_length * config.aggregation_count
        return len(record) * config.num_needles, len(prompt) + answer_len
    raise ValueError(f"unknown RULER-lite task: {config.task}")
