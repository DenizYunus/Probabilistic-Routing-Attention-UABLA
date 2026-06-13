"""Byte-level language-modeling data and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = object
    Dataset = object

from .experiment import (
    _autocast_context,
    _diagnostics_from_output,
    _peak_memory_mb,
    collate_batch,
    infinite_loader,
    language_model_loss,
    move_batch,
)
from .model import TinyTransformerLM

BYTE_PAD_TOKEN = 256
BYTE_BOS_TOKEN = 257
BYTE_EOS_TOKEN = 258
BYTE_LM_VOCAB_SIZE = 259

DEFAULT_BYTE_CORPUS = (
    "UABLA routes compact byte memories instead of attending everywhere. "
    "Dense attention is a strong teacher, but byte streams make length expensive. "
    "A tokenizer-free model should learn local spelling and long-range recall together. "
).encode("utf-8")


@dataclass(frozen=True)
class ByteLMConfig:
    """Configuration for fixed-length next-byte LM samples."""

    seq_len: int = 256
    dataset_size: int = 4096
    seed: int = 0

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")


@dataclass(frozen=True)
class ByteNeedleRecallConfig:
    """Configuration for byte-level long-range needle recall."""

    seq_len: int = 1024
    dataset_size: int = 4096
    seed: int = 0
    code_length: int = 12
    min_gap: int = 512

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if self.code_length <= 0:
            raise ValueError("code_length must be positive")
        if self.min_gap <= 0:
            raise ValueError("min_gap must be positive")
        source_len = len(NEEDLE_SOURCE_PREFIX) + self.code_length + 1
        answer_len = len(NEEDLE_ANSWER_PROMPT) + self.code_length
        if self.seq_len + 1 < source_len + self.min_gap + answer_len:
            raise ValueError("seq_len is too short for source, gap, and answer")


@dataclass(frozen=True)
class ByteLMMetrics:
    loss: float
    lm_loss: float
    byte_accuracy: float
    byte_perplexity: float
    answer_loss: float | None
    answer_accuracy: float | None
    answer_random_chance: float | None
    tokens_per_second: float
    cache_dim_per_token_per_layer: int
    peak_memory_mb: float | None
    diagnostics: dict[str, float]


NEEDLE_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NEEDLE_SOURCE_PREFIX = b"\nSECRET_CODE: "
NEEDLE_ANSWER_PROMPT = b"\nQUESTION: What is the SECRET_CODE?\nANSWER: "


class ByteLanguageModelingDataset(Dataset):
    """Sample deterministic fixed-length next-byte windows from a byte stream."""

    def __init__(self, byte_ids: torch.Tensor, config: ByteLMConfig) -> None:
        if torch is None:
            raise ModuleNotFoundError("ByteLanguageModelingDataset requires torch")
        if byte_ids.ndim != 1:
            raise ValueError("byte_ids must be a 1D tensor")
        if byte_ids.numel() < config.seq_len + 1:
            raise ValueError("byte stream must contain at least seq_len + 1 bytes")
        if int(byte_ids.min().item()) < 0 or int(byte_ids.max().item()) > 255:
            raise ValueError("raw byte IDs must be in [0, 255]")
        self.byte_ids = byte_ids.to(dtype=torch.long).cpu()
        self.config = config

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + index)
        max_start = self.byte_ids.numel() - self.config.seq_len - 1
        if max_start == 0:
            start = 0
        else:
            start = int(
                torch.randint(max_start + 1, (1,), generator=generator, dtype=torch.long).item(),
            )
        window = self.byte_ids[start : start + self.config.seq_len + 1]
        return {
            "input_ids": window[:-1].clone(),
            "labels": window[1:].clone(),
        }


class ByteNeedleRecallDataset(Dataset):
    """Inject a random byte code and ask for it after a long byte gap."""

    def __init__(self, byte_ids: torch.Tensor, config: ByteNeedleRecallConfig) -> None:
        if torch is None:
            raise ModuleNotFoundError("ByteNeedleRecallDataset requires torch")
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
        code = self._sample_code(generator)

        source = torch.cat(
            [
                bytes_to_ids(NEEDLE_SOURCE_PREFIX),
                code,
                torch.tensor([ord("\n")], dtype=torch.long),
            ],
        )
        answer_prompt = bytes_to_ids(NEEDLE_ANSWER_PROMPT)
        answer_segment = torch.cat([answer_prompt, code])
        answer_segment_start = full_len - answer_segment.numel()
        latest_source_start = answer_segment_start - config.min_gap - source.numel()
        if latest_source_start < 0:
            raise RuntimeError("invalid needle layout; config validation should have caught this")
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

        source_code_start = source_start + len(NEEDLE_SOURCE_PREFIX)
        answer_code_start = answer_segment_start + answer_prompt.numel()
        answer_positions = torch.arange(
            answer_code_start,
            answer_code_start + config.code_length,
            dtype=torch.long,
        )
        source_positions = torch.arange(
            source_code_start,
            source_code_start + config.code_length,
            dtype=torch.long,
        )

        return {
            "input_ids": tokens[:-1].clone(),
            "labels": tokens[1:].clone(),
            "answer_index": answer_positions - 1,
            "answer_token": code.clone(),
            "answer_source_index": source_positions.clone(),
            "needle_gap": torch.tensor(answer_code_start - source_code_start, dtype=torch.long),
            "answer_random_chance": torch.tensor(
                1.0 / float(len(NEEDLE_ALPHABET)),
                dtype=torch.float32,
            ),
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

    def _sample_code(self, generator: torch.Generator) -> torch.Tensor:
        alphabet = torch.tensor(list(NEEDLE_ALPHABET), dtype=torch.long)
        indices = torch.randint(
            len(NEEDLE_ALPHABET),
            (self.config.code_length,),
            generator=generator,
            dtype=torch.long,
        )
        return alphabet[indices]


def bytes_to_ids(data: bytes) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError("bytes_to_ids requires torch")
    return torch.tensor(list(data), dtype=torch.long)


def ids_to_bytes(ids: torch.Tensor) -> bytes:
    if torch is None:
        raise ModuleNotFoundError("ids_to_bytes requires torch")
    values = [int(value) for value in ids.reshape(-1).tolist()]
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("only raw byte IDs in [0, 255] can be decoded")
    return bytes(values)


def ensure_min_bytes(data: bytes, min_length: int) -> bytes:
    """Repeat a byte string until it is long enough for deterministic sampling."""

    if min_length <= 0:
        raise ValueError("min_length must be positive")
    if not data:
        raise ValueError("byte corpus cannot be empty")
    repeats = math.ceil(min_length / len(data))
    return (data * repeats)[: max(min_length, len(data))]


def split_byte_stream(
    byte_ids: torch.Tensor,
    *,
    seq_len: int,
    eval_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split bytes into train/eval streams while keeping each side sampleable."""

    if not 0.0 < eval_fraction < 0.5:
        raise ValueError("eval_fraction must be between 0 and 0.5")
    min_side = seq_len + 1
    if byte_ids.numel() < min_side * 2:
        raise ValueError("byte stream is too short to split")
    split = int(byte_ids.numel() * (1.0 - eval_fraction))
    split = min(max(split, min_side), byte_ids.numel() - min_side)
    return byte_ids[:split].clone(), byte_ids[split:].clone()


def make_byte_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    if torch is None:
        raise ModuleNotFoundError("make_byte_loader requires torch")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )


def train_byte_lm_steps(
    model: TinyTransformerLM,
    loader: DataLoader,
    *,
    steps: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    lm_loss_weight: float = 1.0,
    answer_loss_weight: float = 0.0,
    diagnostics_every: int = 0,
    log_every: int = 10,
    metrics_callback: Callable[[ByteLMMetrics], None] | None = None,
) -> list[ByteLMMetrics]:
    if torch is None:
        raise ModuleNotFoundError("train_byte_lm_steps requires torch")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if diagnostics_every < 0:
        raise ValueError("diagnostics_every cannot be negative")
    if amp_dtype is None:
        amp_dtype = torch.float16
    model.train()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp and device.type == "cuda" and amp_dtype == torch.float16,
    )
    stream = infinite_loader(loader)
    metrics: list[ByteLMMetrics] = []
    start_time = time.perf_counter()
    token_count = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        last_batch = None
        last_logits = None
        last_loss = None
        last_lm_loss = None
        last_answer_loss = None
        for _ in range(grad_accum_steps):
            batch = move_batch(next(stream), device)
            with _autocast_context(device, enabled=amp, dtype=amp_dtype):
                output = model(
                    batch["input_ids"],
                    return_routing=False,
                    return_attention=False,
                )
                lm_loss = language_model_loss(output.logits, batch["labels"])
                loss = lm_loss_weight * lm_loss
                answer_loss = None
                if "answer_index" in batch:
                    answer_loss = byte_answer_cross_entropy(output.logits, batch)
                    if answer_loss_weight > 0:
                        loss = loss + answer_loss_weight * answer_loss
            scaler.scale(loss / grad_accum_steps).backward()
            token_count += batch["input_ids"].numel()
            last_batch = batch
            last_logits = output.logits.detach()
            last_loss = loss.detach()
            last_lm_loss = lm_loss.detach()
            last_answer_loss = answer_loss.detach() if answer_loss is not None else None

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0 or step == steps:
            if last_batch is None or last_logits is None or last_loss is None or last_lm_loss is None:
                raise RuntimeError("training step did not produce metrics")
            diagnostic_output = None
            if _should_collect_diagnostics(model, step, diagnostics_every):
                diagnostic_output = _diagnostic_forward(
                    model,
                    last_batch,
                    device=device,
                    amp=amp,
                    amp_dtype=amp_dtype,
                )
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            metrics.append(
                _collect_byte_metrics(
                    model,
                    last_batch,
                    loss=last_loss,
                    lm_loss=last_lm_loss,
                    answer_loss=last_answer_loss,
                    logits=last_logits,
                    tokens_per_second=token_count / elapsed,
                    diagnostic_output=diagnostic_output,
                )
            )
            if metrics_callback is not None:
                metrics_callback(metrics[-1])
    return metrics


def evaluate_byte_lm(
    model: TinyTransformerLM,
    loader: DataLoader,
    *,
    device: torch.device,
    batches: int,
    amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    collect_diagnostics: bool = True,
) -> ByteLMMetrics:
    if torch is None:
        raise ModuleNotFoundError("evaluate_byte_lm requires torch")
    if amp_dtype is None:
        amp_dtype = torch.float16
    model.eval()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_accuracy = 0.0
    total_answer_loss = 0.0
    total_answer_accuracy = 0.0
    answer_batches = 0
    answer_random_chance = None
    total_tokens = 0
    diagnostics: dict[str, float] = {}
    start_time = time.perf_counter()
    last_output = None
    last_batch = None
    seen_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= batches:
                break
            batch = move_batch(batch, device)
            with _autocast_context(device, enabled=amp, dtype=amp_dtype):
                output = model(
                    batch["input_ids"],
                    return_routing=False,
                    return_attention=False,
                )
                loss = language_model_loss(output.logits, batch["labels"])
                accuracy = byte_accuracy(output.logits, batch["labels"])
                if "answer_index" in batch:
                    ans_loss = byte_answer_cross_entropy(output.logits, batch)
                    ans_acc = byte_answer_accuracy(output.logits, batch)
                else:
                    ans_loss = None
                    ans_acc = None
            total_loss += float(loss.item())
            total_lm_loss += float(loss.item())
            total_accuracy += float(accuracy.item())
            if ans_loss is not None and ans_acc is not None:
                total_answer_loss += float(ans_loss.item())
                total_answer_accuracy += float(ans_acc.item())
                answer_batches += 1
                answer_random_chance = byte_answer_random_chance(batch)
            total_tokens += batch["input_ids"].numel()
            last_output = None
            last_batch = batch
            seen_batches += 1
    count = max(seen_batches, 1)
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    if collect_diagnostics and last_batch is not None:
        last_output = _diagnostic_forward(
            model,
            last_batch,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
        )
    if last_output is not None and last_output.uabla_outputs:
        diagnostics = _diagnostics_from_output(last_output)
    avg_loss = total_loss / count
    avg_lm_loss = total_lm_loss / count
    return ByteLMMetrics(
        loss=avg_loss,
        lm_loss=avg_lm_loss,
        byte_accuracy=total_accuracy / count,
        byte_perplexity=_safe_perplexity(avg_lm_loss),
        answer_loss=total_answer_loss / answer_batches if answer_batches else None,
        answer_accuracy=total_answer_accuracy / answer_batches if answer_batches else None,
        answer_random_chance=answer_random_chance,
        tokens_per_second=total_tokens / elapsed,
        cache_dim_per_token_per_layer=model.cache_dim_per_token_per_layer,
        peak_memory_mb=_peak_memory_mb(device),
        diagnostics=diagnostics,
    )


def byte_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean()


def byte_answer_cross_entropy(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    answer_logits, answer_targets = _gather_byte_answer_logits(logits, batch)
    return torch.nn.functional.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_targets.reshape(-1),
    )


def byte_answer_accuracy(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    answer_logits, answer_targets = _gather_byte_answer_logits(logits, batch)
    predictions = answer_logits.argmax(dim=-1)
    return (predictions == answer_targets).float().mean()


def byte_answer_random_chance(batch: dict[str, torch.Tensor]) -> float | None:
    if "answer_random_chance" not in batch:
        return None
    return float(batch["answer_random_chance"][0].item())


def _collect_byte_metrics(
    model: TinyTransformerLM,
    batch: dict[str, torch.Tensor],
    *,
    loss: torch.Tensor,
    lm_loss: torch.Tensor,
    answer_loss: torch.Tensor | None,
    logits: torch.Tensor,
    tokens_per_second: float,
    diagnostic_output,
) -> ByteLMMetrics:
    loss_value = float(loss.item())
    lm_loss_value = float(lm_loss.item())
    answer_accuracy = (
        float(byte_answer_accuracy(logits, batch).item()) if "answer_index" in batch else None
    )
    return ByteLMMetrics(
        loss=loss_value,
        lm_loss=lm_loss_value,
        byte_accuracy=float(byte_accuracy(logits, batch["labels"]).item()),
        byte_perplexity=_safe_perplexity(lm_loss_value),
        answer_loss=float(answer_loss.item()) if answer_loss is not None else None,
        answer_accuracy=answer_accuracy,
        answer_random_chance=byte_answer_random_chance(batch),
        tokens_per_second=tokens_per_second,
        cache_dim_per_token_per_layer=model.cache_dim_per_token_per_layer,
        peak_memory_mb=_peak_memory_mb(batch["input_ids"].device),
        diagnostics=_diagnostics_from_output(diagnostic_output) if diagnostic_output else {},
    )


def _gather_byte_answer_logits(
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
    raise ValueError("answer_index must have shape [batch] or [batch, answer_bytes]")


def _should_collect_diagnostics(
    model: TinyTransformerLM,
    step: int,
    diagnostics_every: int,
) -> bool:
    return model.attention_type == "uabla" and diagnostics_every > 0 and step % diagnostics_every == 0


def _diagnostic_forward(
    model: TinyTransformerLM,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_context(device, enabled=amp, dtype=amp_dtype):
            return model(batch["input_ids"], return_routing=True, return_attention=False)
    finally:
        if was_training:
            model.train()


def _safe_perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))
