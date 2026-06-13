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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    ).to(device)
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
                "lm_loss_weight": args.lm_loss_weight,
                "answer_loss_weight": args.answer_loss_weight,
                "diagnostics_every": args.diagnostics_every,
                "eval_diagnostics": not args.no_eval_diagnostics,
                "log_every": args.log_every,
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
