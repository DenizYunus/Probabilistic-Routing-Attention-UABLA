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
class ByteLMMetrics:
    loss: float
    byte_accuracy: float
    byte_perplexity: float
    tokens_per_second: float
    cache_dim_per_token_per_layer: int
    peak_memory_mb: float | None
    diagnostics: dict[str, float]


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
    dataset: ByteLanguageModelingDataset,
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
    log_every: int = 10,
    metrics_callback: Callable[[ByteLMMetrics], None] | None = None,
) -> list[ByteLMMetrics]:
    if torch is None:
        raise ModuleNotFoundError("train_byte_lm_steps requires torch")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
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
        last_output = None
        last_loss = None
        for _ in range(grad_accum_steps):
            batch = move_batch(next(stream), device)
            with _autocast_context(device, enabled=amp, dtype=amp_dtype):
                output = model(
                    batch["input_ids"],
                    return_routing=model.attention_type == "uabla",
                    return_attention=False,
                )
                loss = language_model_loss(output.logits, batch["labels"])
            scaler.scale(loss / grad_accum_steps).backward()
            token_count += batch["input_ids"].numel()
            last_batch = batch
            last_output = output
            last_loss = loss.detach()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0 or step == steps:
            if last_batch is None or last_output is None or last_loss is None:
                raise RuntimeError("training step did not produce metrics")
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            metrics.append(
                _collect_byte_metrics(
                    model,
                    last_batch,
                    loss=last_loss,
                    tokens_per_second=token_count / elapsed,
                    output=last_output,
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
) -> ByteLMMetrics:
    if torch is None:
        raise ModuleNotFoundError("evaluate_byte_lm requires torch")
    if amp_dtype is None:
        amp_dtype = torch.float16
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0
    diagnostics: dict[str, float] = {}
    start_time = time.perf_counter()
    last_output = None
    seen_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= batches:
                break
            batch = move_batch(batch, device)
            with _autocast_context(device, enabled=amp, dtype=amp_dtype):
                output = model(
                    batch["input_ids"],
                    return_routing=model.attention_type == "uabla",
                    return_attention=False,
                )
                loss = language_model_loss(output.logits, batch["labels"])
                accuracy = byte_accuracy(output.logits, batch["labels"])
            total_loss += float(loss.item())
            total_accuracy += float(accuracy.item())
            total_tokens += batch["input_ids"].numel()
            last_output = output
            seen_batches += 1
    count = max(seen_batches, 1)
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    if last_output is not None and last_output.uabla_outputs:
        diagnostics = _diagnostics_from_output(last_output)
    avg_loss = total_loss / count
    return ByteLMMetrics(
        loss=avg_loss,
        byte_accuracy=total_accuracy / count,
        byte_perplexity=_safe_perplexity(avg_loss),
        tokens_per_second=total_tokens / elapsed,
        cache_dim_per_token_per_layer=model.cache_dim_per_token_per_layer,
        peak_memory_mb=_peak_memory_mb(device),
        diagnostics=diagnostics,
    )


def byte_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels).float().mean()


def _collect_byte_metrics(
    model: TinyTransformerLM,
    batch: dict[str, torch.Tensor],
    *,
    loss: torch.Tensor,
    tokens_per_second: float,
    output,
) -> ByteLMMetrics:
    loss_value = float(loss.item())
    return ByteLMMetrics(
        loss=loss_value,
        byte_accuracy=float(byte_accuracy(output.logits.detach(), batch["labels"]).item()),
        byte_perplexity=_safe_perplexity(loss_value),
        tokens_per_second=tokens_per_second,
        cache_dim_per_token_per_layer=model.cache_dim_per_token_per_layer,
        peak_memory_mb=_peak_memory_mb(batch["input_ids"].device),
        diagnostics=_diagnostics_from_output(output),
    )


def _safe_perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))
