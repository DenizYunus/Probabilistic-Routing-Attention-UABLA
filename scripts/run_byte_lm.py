#!/usr/bin/env python3
"""Run tokenizer-free byte-level LM experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from uabla.byte_lm import (
    BYTE_LM_VOCAB_SIZE,
    DEFAULT_BYTE_CORPUS,
    ByteLMConfig,
    ByteLMMetrics,
    ByteLanguageModelingDataset,
    ByteNeedleRecallConfig,
    ByteNeedleRecallDataset,
    RoutingBudgetStage,
    bytes_to_ids,
    ensure_min_bytes,
    evaluate_byte_lm,
    make_byte_loader,
    split_byte_stream,
    train_byte_lm_steps,
)
from uabla.config import UABLAConfig
from uabla.model import TinyTransformerLM


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def amp_dtype_from_arg(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown AMP dtype: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", choices=["uabla", "dense", "local"], default="uabla")
    parser.add_argument("--task", choices=["lm", "needle"], default="lm")
    parser.add_argument("--byte-mode", choices=["raw"], default="raw")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--min-corpus-bytes", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--byte-mixer-kernel", type=int, default=5)
    parser.add_argument("--byte-route-patch-size", type=int, default=16)
    parser.add_argument("--routed-span-left", type=int, default=2)
    parser.add_argument("--routed-span-right", type=int, default=8)
    parser.add_argument("--no-shifted-routing-blocks", action="store_true")
    parser.add_argument("--shifted-block-offset", type=int)
    parser.add_argument("--stage-shifted-after-steps", type=int)
    parser.add_argument("--stage-shifted-when-answer-accuracy", type=float)
    parser.add_argument("--stage-shifted-min-step", type=int, default=1)
    parser.add_argument("--needle-code-length", type=int, default=12)
    parser.add_argument("--needle-min-gap", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--diagnostics-every", type=int, default=0)
    parser.add_argument("--no-eval-diagnostics", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--lm-loss-weight", type=float, default=1.0)
    parser.add_argument("--answer-loss-weight", type=float, default=0.0)
    parser.add_argument("--no-vectorized-routing", action="store_true")
    parser.add_argument("--route-budget-curriculum", choices=["none", "explore"], default="none")
    parser.add_argument("--route-budget-curriculum-boundaries", default="2500,6500")
    parser.add_argument("--route-entropy-weight", type=float, default=0.0)
    parser.add_argument("--route-entropy-min", type=float, default=0.35)
    parser.add_argument("--route-entropy-warmup-steps", type=int, default=0)
    parser.add_argument("--route-entropy-decay-steps", type=int, default=0)
    parser.add_argument("--direct-route-weight", type=float, default=0.0)
    parser.add_argument("--direct-route-warmup-steps", type=int, default=0)
    parser.add_argument("--direct-route-decay-steps", type=int, default=0)
    parser.add_argument("--token-contrast-weight", type=float, default=0.0)
    parser.add_argument("--token-contrast-warmup-steps", type=int, default=0)
    parser.add_argument("--token-contrast-decay-steps", type=int, default=0)
    parser.add_argument("--token-contrast-temperature", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    return parser.parse_args()


def build_route_budget_stages(args: argparse.Namespace) -> tuple[RoutingBudgetStage, ...]:
    if args.route_budget_curriculum == "none":
        return ()
    if args.route_budget_curriculum != "explore":
        raise ValueError(f"unknown route budget curriculum: {args.route_budget_curriculum}")

    boundaries = parse_curriculum_boundaries(args.route_budget_curriculum_boundaries)
    first_medium_step = boundaries[0] + 1
    first_final_step = boundaries[1] + 1
    return (
        RoutingBudgetStage(
            start_step=1,
            superblock_hit_buckets=(4, 8),
            centroid_hit_buckets=(16, 32),
            token_k_buckets=(32, 64),
        ),
        RoutingBudgetStage(
            start_step=first_medium_step,
            superblock_hit_buckets=(2, 4),
            centroid_hit_buckets=(8, 16),
            token_k_buckets=(16, 32),
        ),
        RoutingBudgetStage(
            start_step=first_final_step,
            superblock_hit_buckets=(2, 4),
            centroid_hit_buckets=(4, 8),
            token_k_buckets=(8, 16),
        ),
    )


def parse_curriculum_boundaries(value: str) -> tuple[int, int]:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--route-budget-curriculum-boundaries must be two integers") from exc
    if len(parts) != 2:
        raise ValueError("--route-budget-curriculum-boundaries must contain two comma-separated steps")
    if parts[0] <= 0 or parts[1] <= 0:
        raise ValueError("route budget curriculum boundaries must be positive")
    if parts[0] >= parts[1]:
        raise ValueError("route budget curriculum boundaries must be strictly increasing")
    return parts[0], parts[1]


def budget_stage_to_json(stage: RoutingBudgetStage) -> dict[str, Any]:
    return {
        "centroid_hit_buckets": list(stage.centroid_hit_buckets),
        "start_step": stage.start_step,
        "superblock_hit_buckets": (
            list(stage.superblock_hit_buckets) if stage.superblock_hit_buckets is not None else None
        ),
        "token_k_buckets": list(stage.token_k_buckets),
    }


def main() -> None:
    args = parse_args()
    budget_stages = build_route_budget_stages(args)
    stage_shifted = (
        args.stage_shifted_after_steps is not None
        or args.stage_shifted_when_answer_accuracy is not None
    )
    if stage_shifted:
        if args.attention != "uabla":
            raise ValueError("shifted routing stages require --attention uabla")
        if args.no_shifted_routing_blocks:
            raise ValueError("shifted routing stages conflict with --no-shifted-routing-blocks")
    if args.stage_shifted_after_steps is not None:
        if args.stage_shifted_after_steps < 0:
            raise ValueError("--stage-shifted-after-steps cannot be negative")
        if args.stage_shifted_after_steps >= args.steps:
            raise ValueError("--stage-shifted-after-steps must be smaller than --steps")
    if args.stage_shifted_when_answer_accuracy is not None:
        if not 0.0 <= args.stage_shifted_when_answer_accuracy <= 1.0:
            raise ValueError("--stage-shifted-when-answer-accuracy must be in [0, 1]")
    if args.stage_shifted_min_step <= 0:
        raise ValueError("--stage-shifted-min-step must be positive")
    if budget_stages and args.attention != "uabla":
        raise ValueError("--route-budget-curriculum requires --attention uabla")
    if args.route_entropy_weight < 0:
        raise ValueError("--route-entropy-weight cannot be negative")
    if args.route_entropy_weight > 0 and args.attention != "uabla":
        raise ValueError("--route-entropy-weight requires --attention uabla")
    if not 0.0 <= args.route_entropy_min <= 1.0:
        raise ValueError("--route-entropy-min must be in [0, 1]")
    if args.route_entropy_warmup_steps < 0 or args.route_entropy_decay_steps < 0:
        raise ValueError("route entropy schedule steps must be non-negative")
    if args.direct_route_weight < 0:
        raise ValueError("--direct-route-weight cannot be negative")
    if args.direct_route_weight > 0 and args.attention != "uabla":
        raise ValueError("--direct-route-weight requires --attention uabla")
    if args.direct_route_warmup_steps < 0 or args.direct_route_decay_steps < 0:
        raise ValueError("direct route schedule steps must be non-negative")
    if args.token_contrast_weight < 0:
        raise ValueError("--token-contrast-weight cannot be negative")
    if args.token_contrast_weight > 0 and args.attention != "uabla":
        raise ValueError("--token-contrast-weight requires --attention uabla")
    if args.token_contrast_warmup_steps < 0 or args.token_contrast_decay_steps < 0:
        raise ValueError("token contrast schedule steps must be non-negative")
    if args.token_contrast_temperature <= 0:
        raise ValueError("--token-contrast-temperature must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    raw_bytes = args.text_file.read_bytes() if args.text_file is not None else DEFAULT_BYTE_CORPUS
    min_bytes = max(args.min_corpus_bytes, 2 * (args.seq_len + 1))
    byte_ids = bytes_to_ids(ensure_min_bytes(raw_bytes, min_bytes))
    train_bytes, eval_bytes = split_byte_stream(
        byte_ids,
        seq_len=args.seq_len,
        eval_fraction=args.eval_fraction,
    )

    if args.task == "lm":
        train_config = ByteLMConfig(
            seq_len=args.seq_len,
            dataset_size=args.train_size,
            seed=args.seed,
        )
        eval_config = ByteLMConfig(
            seq_len=args.seq_len,
            dataset_size=args.eval_size,
            seed=args.seed + 10_000,
        )
        train_dataset = ByteLanguageModelingDataset(train_bytes, train_config)
        eval_dataset = ByteLanguageModelingDataset(eval_bytes, eval_config)
    else:
        train_config = ByteNeedleRecallConfig(
            seq_len=args.seq_len,
            dataset_size=args.train_size,
            seed=args.seed,
            code_length=args.needle_code_length,
            min_gap=args.needle_min_gap,
        )
        eval_config = ByteNeedleRecallConfig(
            seq_len=args.seq_len,
            dataset_size=args.eval_size,
            seed=args.seed + 10_000,
            code_length=args.needle_code_length,
            min_gap=args.needle_min_gap,
        )
        train_dataset = ByteNeedleRecallDataset(train_bytes, train_config)
        eval_dataset = ByteNeedleRecallDataset(eval_bytes, eval_config)
    train_loader = make_byte_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = make_byte_loader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    uabla_config = None
    if args.attention == "uabla":
        uabla_config = UABLAConfig(
            hidden_size=args.hidden_size,
            routing_dim=max(8, args.hidden_size // 4),
            value_dim=max(16, args.hidden_size // 2),
            position_dim=max(4, args.hidden_size // 8),
            block_size=args.byte_route_patch_size,
            local_window=args.local_window,
            centroid_hit_buckets=(4, 8),
            token_k_buckets=(8, 16),
            superblock_size_blocks=4,
            superblock_hit_buckets=(2, 4),
            vectorized_routing=not args.no_vectorized_routing,
            routed_span_left=args.routed_span_left,
            routed_span_right=args.routed_span_right,
            use_shifted_blocks=False if stage_shifted else not args.no_shifted_routing_blocks,
            shifted_block_offset=args.shifted_block_offset,
        )
    model = TinyTransformerLM(
        vocab_size=BYTE_LM_VOCAB_SIZE,
        max_seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        attention_type=args.attention,
        local_window=args.local_window,
        uabla_config=uabla_config,
        input_mixer_kernel=args.byte_mixer_kernel,
    )
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_index = 0

    def emit_train_metrics(metrics: ByteLMMetrics) -> None:
        nonlocal log_index
        log_index += 1
        print(
            json.dumps(
                {
                    "phase": "train",
                    "log_index": log_index,
                    **metrics.__dict__,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def emit_stage_event(event: dict[str, object]) -> None:
        print(json.dumps({"phase": "stage", **event}, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "phase": "start",
                "attention": args.attention,
                "task": args.task,
                "byte_mode": args.byte_mode,
                "byte_mixer_kernel": args.byte_mixer_kernel,
                "byte_route_patch_size": args.byte_route_patch_size,
                "routed_span_left": args.routed_span_left if args.attention == "uabla" else None,
                "routed_span_right": args.routed_span_right if args.attention == "uabla" else None,
                "shifted_routing_blocks": (
                    not args.no_shifted_routing_blocks if args.attention == "uabla" else None
                ),
                "shifted_routing_blocks_initial": (
                    uabla_config.use_shifted_blocks if uabla_config is not None else None
                ),
                "shifted_block_offset": (
                    uabla_config.shifted_block_offset_value
                    if uabla_config is not None
                    and (uabla_config.use_shifted_blocks or stage_shifted)
                    else None
                ),
                "stage_shifted_after_steps": args.stage_shifted_after_steps,
                "stage_shifted_min_step": args.stage_shifted_min_step,
                "stage_shifted_when_answer_accuracy": args.stage_shifted_when_answer_accuracy,
                "needle_code_length": args.needle_code_length if args.task == "needle" else None,
                "needle_min_gap": args.needle_min_gap if args.task == "needle" else None,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "seq_len": args.seq_len,
                "vocab_size": BYTE_LM_VOCAB_SIZE,
                "raw_corpus_bytes": len(raw_bytes),
                "train_stream_bytes": int(train_bytes.numel()),
                "eval_stream_bytes": int(eval_bytes.numel()),
                "device": str(device),
                "amp": args.amp,
                "amp_dtype": args.amp_dtype,
                "grad_accum_steps": args.grad_accum_steps,
                "init_checkpoint": (
                    str(args.init_checkpoint) if args.init_checkpoint is not None else None
                ),
                "lm_loss_weight": args.lm_loss_weight,
                "answer_loss_weight": args.answer_loss_weight,
                "diagnostics_every": args.diagnostics_every,
                "eval_diagnostics": not args.no_eval_diagnostics,
                "log_every": args.log_every,
                "route_budget_curriculum": args.route_budget_curriculum,
                "route_budget_curriculum_boundaries": args.route_budget_curriculum_boundaries,
                "route_budget_stages": [budget_stage_to_json(stage) for stage in budget_stages],
                "route_entropy_decay_steps": args.route_entropy_decay_steps,
                "route_entropy_min": args.route_entropy_min,
                "route_entropy_warmup_steps": args.route_entropy_warmup_steps,
                "route_entropy_weight": args.route_entropy_weight,
                "direct_route_decay_steps": args.direct_route_decay_steps,
                "direct_route_warmup_steps": args.direct_route_warmup_steps,
                "direct_route_weight": args.direct_route_weight,
                "token_contrast_decay_steps": args.token_contrast_decay_steps,
                "token_contrast_temperature": args.token_contrast_temperature,
                "token_contrast_warmup_steps": args.token_contrast_warmup_steps,
                "token_contrast_weight": args.token_contrast_weight,
            },
            sort_keys=True,
        ),
        flush=True,
        file=sys.stderr,
    )

    train_byte_lm_steps(
        model,
        train_loader,
        steps=args.steps,
        optimizer=optimizer,
        device=device,
        amp=args.amp,
        amp_dtype=amp_dtype_from_arg(args.amp_dtype),
        lm_loss_weight=args.lm_loss_weight,
        answer_loss_weight=args.answer_loss_weight,
        diagnostics_every=args.diagnostics_every,
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        shifted_enable_step=args.stage_shifted_after_steps,
        shifted_enable_answer_accuracy=args.stage_shifted_when_answer_accuracy,
        shifted_enable_min_step=args.stage_shifted_min_step,
        route_budget_stages=budget_stages,
        route_entropy_weight=args.route_entropy_weight,
        route_entropy_min=args.route_entropy_min,
        route_entropy_warmup_steps=args.route_entropy_warmup_steps,
        route_entropy_decay_steps=args.route_entropy_decay_steps,
        direct_route_weight=args.direct_route_weight,
        direct_route_warmup_steps=args.direct_route_warmup_steps,
        direct_route_decay_steps=args.direct_route_decay_steps,
        token_contrast_weight=args.token_contrast_weight,
        token_contrast_warmup_steps=args.token_contrast_warmup_steps,
        token_contrast_decay_steps=args.token_contrast_decay_steps,
        token_contrast_temperature=args.token_contrast_temperature,
        stage_callback=emit_stage_event,
        metrics_callback=emit_train_metrics,
    )

    eval_metrics = evaluate_byte_lm(
        model,
        eval_loader,
        device=device,
        batches=args.eval_batches,
        amp=args.amp,
        amp_dtype=amp_dtype_from_arg(args.amp_dtype),
        collect_diagnostics=not args.no_eval_diagnostics,
    )
    print(json.dumps({"phase": "eval", **eval_metrics.__dict__}, sort_keys=True), flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "attention": args.attention,
                "config": json_safe(vars(args)),
            },
            args.save_checkpoint,
        )


if __name__ == "__main__":
    main()
