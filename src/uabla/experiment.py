"""Experiment utilities for tiny UABLA synthetic comparisons."""

from __future__ import annotations

from contextlib import nullcontext
import time
from dataclasses import dataclass
from typing import Callable, Iterable

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None
    DataLoader = object

from .losses import (
    direct_routing_supervision_loss,
    guarded_budget_loss,
    routing_diagnostics,
    routing_distillation_loss,
)
from .model import TinyTransformerLM
from .synthetic import KeyValueRecallDataset, answer_accuracy, answer_cross_entropy
from .synthetic import value_answer_accuracy, value_answer_cross_entropy, value_random_chance


@dataclass(frozen=True)
class ExperimentMetrics:
    loss: float
    answer_loss: float
    answer_accuracy: float
    value_answer_loss: float
    value_answer_accuracy: float
    value_random_chance: float
    tokens_per_second: float
    cache_dim_per_token_per_layer: int
    peak_memory_mb: float | None
    diagnostics: dict[str, float]


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError("collate_batch requires torch")
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def infinite_loader(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def make_loader(
    dataset: KeyValueRecallDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    if torch is None:
        raise ModuleNotFoundError("make_loader requires torch")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def language_model_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def train_steps(
    model: TinyTransformerLM,
    loader: DataLoader,
    *,
    steps: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher: TinyTransformerLM | None = None,
    lm_loss_weight: float = 0.2,
    answer_loss_weight: float = 1.0,
    distill_weight: float = 0.0,
    direct_route_weight: float = 0.0,
    budget_weight: float = 0.0,
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    log_every: int = 10,
    metrics_callback: Callable[[ExperimentMetrics], None] | None = None,
) -> list[ExperimentMetrics]:
    """Train for a small number of steps and return periodic metrics."""

    if torch is None:
        raise ModuleNotFoundError("train_steps requires torch")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if amp_dtype is None:
        amp_dtype = torch.float16
    model.train()
    if teacher is not None:
        teacher.eval()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp and device.type == "cuda" and amp_dtype == torch.float16,
    )
    stream = infinite_loader(loader)
    metrics: list[ExperimentMetrics] = []
    start_time = time.perf_counter()
    token_count = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        last_batch = None
        last_output = None
        last_loss = None
        last_ans_loss = None
        for _ in range(grad_accum_steps):
            batch = move_batch(next(stream), device)
            needs_aux = model.attention_type == "uabla"
            with _autocast_context(device, enabled=amp, dtype=amp_dtype):
                output = model(
                    batch["input_ids"],
                    return_routing=needs_aux,
                    return_attention=False,
                )
                lm_loss = language_model_loss(output.logits, batch["labels"])
                ans_loss = answer_cross_entropy(output.logits, batch)
                loss = lm_loss_weight * lm_loss + answer_loss_weight * ans_loss

                if teacher is not None and distill_weight > 0 and output.uabla_outputs:
                    with torch.no_grad():
                        teacher_output = teacher(batch["input_ids"], return_attention=True)
                    distill_losses = []
                    for uabla_layer, teacher_attention in zip(
                        output.uabla_outputs,
                        teacher_output.dense_attentions,
                        strict=False,
                    ):
                        if (
                            uabla_layer.route_scores is not None
                            and uabla_layer.routeable_mask is not None
                        ):
                            distill_losses.append(
                                routing_distillation_loss(
                                    uabla_layer.route_scores,
                                    uabla_layer.routeable_mask,
                                    teacher_attention,
                                    region_size_tokens=model.uabla_config.block_size,
                                )
                            )
                        if (
                            uabla_layer.superblock_route_scores is not None
                            and uabla_layer.superblock_routeable_mask is not None
                        ):
                            distill_losses.append(
                                routing_distillation_loss(
                                    uabla_layer.superblock_route_scores,
                                    uabla_layer.superblock_routeable_mask,
                                    teacher_attention,
                                    region_size_tokens=(
                                        model.uabla_config.block_size
                                        * model.uabla_config.superblock_size_blocks
                                    ),
                                )
                            )
                    if distill_losses:
                        loss = loss + distill_weight * torch.stack(distill_losses).mean()

                if (
                    direct_route_weight > 0
                    and output.uabla_outputs
                    and "answer_source_index" in batch
                ):
                    direct_route_losses = []
                    for uabla_layer in output.uabla_outputs:
                        if (
                            uabla_layer.route_scores is not None
                            and uabla_layer.routeable_mask is not None
                        ):
                            direct_route_losses.append(
                                direct_routing_supervision_loss(
                                    uabla_layer.route_scores,
                                    uabla_layer.routeable_mask,
                                    batch["answer_index"],
                                    batch["answer_source_index"],
                                    region_size_tokens=model.uabla_config.block_size,
                                )
                            )
                        if (
                            uabla_layer.superblock_route_scores is not None
                            and uabla_layer.superblock_routeable_mask is not None
                        ):
                            direct_route_losses.append(
                                direct_routing_supervision_loss(
                                    uabla_layer.superblock_route_scores,
                                    uabla_layer.superblock_routeable_mask,
                                    batch["answer_index"],
                                    batch["answer_source_index"],
                                    region_size_tokens=(
                                        model.uabla_config.block_size
                                        * model.uabla_config.superblock_size_blocks
                                    ),
                                )
                            )
                    if direct_route_losses:
                        loss = loss + direct_route_weight * torch.stack(direct_route_losses).mean()

                if budget_weight > 0 and output.uabla_outputs:
                    budget_losses = [
                        guarded_budget_loss(layer.candidates.token_budgets, logits=output.logits)
                        for layer in output.uabla_outputs
                        if layer.candidates is not None
                    ]
                    if budget_losses:
                        loss = loss + budget_weight * torch.stack(budget_losses).mean()

            scaler.scale(loss / grad_accum_steps).backward()
            token_count += batch["input_ids"].numel()
            last_batch = batch
            last_output = output
            last_loss = loss.detach()
            last_ans_loss = ans_loss.detach()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0 or step == steps:
            if last_batch is None or last_output is None or last_loss is None or last_ans_loss is None:
                raise RuntimeError("training step did not produce metrics")
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            metrics.append(
                _collect_metrics(
                    model,
                    last_batch,
                    loss=last_loss,
                    answer_loss=last_ans_loss,
                    tokens_per_second=token_count / elapsed,
                    output=last_output,
                )
            )
            if metrics_callback is not None:
                metrics_callback(metrics[-1])
    return metrics


def evaluate(
    model: TinyTransformerLM,
    loader: DataLoader,
    *,
    device: torch.device,
    batches: int,
    amp: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> ExperimentMetrics:
    if torch is None:
        raise ModuleNotFoundError("evaluate requires torch")
    if amp_dtype is None:
        amp_dtype = torch.float16
    model.eval()
    total_loss = 0.0
    total_answer_loss = 0.0
    total_answer_accuracy = 0.0
    total_value_answer_loss = 0.0
    total_value_answer_accuracy = 0.0
    random_chance = 0.0
    total_tokens = 0
    diagnostics: dict[str, float] = {}
    start_time = time.perf_counter()
    last_output = None
    last_batch = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= batches:
                break
            batch = move_batch(batch, device)
            needs_aux = model.attention_type == "uabla"
            with _autocast_context(device, enabled=amp, dtype=amp_dtype):
                output = model(
                    batch["input_ids"],
                    return_routing=needs_aux,
                    return_attention=False,
                )
                lm_loss = language_model_loss(output.logits, batch["labels"])
                ans_loss = answer_cross_entropy(output.logits, batch)
                acc = answer_accuracy(output.logits, batch)
                value_ans_loss = value_answer_cross_entropy(output.logits, batch)
                value_acc = value_answer_accuracy(output.logits, batch)
            total_loss += float(lm_loss.item())
            total_answer_loss += float(ans_loss.item())
            total_answer_accuracy += float(acc.item())
            total_value_answer_loss += float(value_ans_loss.item())
            total_value_answer_accuracy += float(value_acc.item())
            random_chance = value_random_chance(batch)
            total_tokens += batch["input_ids"].numel()
            last_output = output
            last_batch = batch
    count = max(min(batches, batch_idx + 1), 1)
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    if last_output is not None and last_output.uabla_outputs:
        diagnostics = _diagnostics_from_output(last_output)
    return ExperimentMetrics(
        loss=total_loss / count,
        answer_loss=total_answer_loss / count,
        answer_accuracy=total_answer_accuracy / count,
        value_answer_loss=total_value_answer_loss / count,
        value_answer_accuracy=total_value_answer_accuracy / count,
        value_random_chance=random_chance,
        tokens_per_second=total_tokens / elapsed,
        cache_dim_per_token_per_layer=model.cache_dim_per_token_per_layer,
        peak_memory_mb=_peak_memory_mb(device),
        diagnostics=diagnostics,
    )


def _collect_metrics(
    model: TinyTransformerLM,
    batch: dict[str, torch.Tensor],
    *,
    loss: torch.Tensor,
    answer_loss: torch.Tensor,
    tokens_per_second: float,
    output,
) -> ExperimentMetrics:
    return ExperimentMetrics(
        loss=float(loss.item()),
        answer_loss=float(answer_loss.item()),
        answer_accuracy=float(answer_accuracy(output.logits.detach(), batch).item()),
        value_answer_loss=float(value_answer_cross_entropy(output.logits.detach(), batch).item()),
        value_answer_accuracy=float(value_answer_accuracy(output.logits.detach(), batch).item()),
        value_random_chance=value_random_chance(batch),
        tokens_per_second=tokens_per_second,
        cache_dim_per_token_per_layer=model.cache_dim_per_token_per_layer,
        peak_memory_mb=_peak_memory_mb(batch["input_ids"].device),
        diagnostics=_diagnostics_from_output(output),
    )


def _diagnostics_from_output(output) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    for layer_idx, layer in enumerate(output.uabla_outputs):
        if layer.candidates is None or layer.route_scores is None or layer.routeable_mask is None:
            continue
        layer_diag = routing_diagnostics(
            layer.candidates,
            layer.route_scores,
            layer.routeable_mask,
            attention=layer.attention,
        )
        for key, value in layer_diag.items():
            diagnostics[f"layer{layer_idx}.{key}"] = float(value.detach().item())
    return diagnostics


def _autocast_context(
    device: torch.device,
    *,
    enabled: bool,
    dtype: torch.dtype,
):
    if not enabled or device.type not in {"cuda", "cpu", "mps"}:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
