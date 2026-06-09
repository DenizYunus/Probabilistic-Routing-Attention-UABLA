#!/usr/bin/env python3
"""Run tiny synthetic UABLA experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from uabla.experiment import evaluate, make_loader, train_steps
from uabla.experiment import ExperimentMetrics
from uabla.model import TinyTransformerLM
from uabla.synthetic import KeyValueRecallConfig, KeyValueRecallDataset


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
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--direct-route-weight", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=0.0)
    parser.add_argument("--lm-loss-weight", type=float, default=0.2)
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
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

    train_config = KeyValueRecallConfig(
        seq_len=args.seq_len,
        num_pairs=args.num_pairs,
        num_queries=args.num_queries,
        dataset_size=args.train_size,
        seed=args.seed,
    )
    eval_config = KeyValueRecallConfig(
        seq_len=args.seq_len,
        num_pairs=args.num_pairs,
        num_queries=args.num_queries,
        dataset_size=args.eval_size,
        seed=args.seed + 10_000,
    )
    train_loader = make_loader(
        KeyValueRecallDataset(train_config),
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = make_loader(
        KeyValueRecallDataset(eval_config),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = TinyTransformerLM(
        vocab_size=train_config.vocab_size,
        max_seq_len=train_config.model_seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        attention_type=args.attention,
        local_window=args.local_window,
        uabla_vectorized_routing=not args.no_vectorized_routing,
    ).to(device)

    teacher = None
    if args.teacher_checkpoint is not None:
        teacher = TinyTransformerLM(
            vocab_size=train_config.vocab_size,
            max_seq_len=train_config.model_seq_len,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            attention_type="dense",
            local_window=args.local_window,
        ).to(device)
        checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(checkpoint["model"])
        teacher.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_index = 0

    def emit_train_metrics(metrics: ExperimentMetrics) -> None:
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
                "steps": args.steps,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "seq_len": args.seq_len,
                "num_pairs": args.num_pairs,
                "num_queries": args.num_queries,
                "device": str(device),
                "amp": args.amp,
                "amp_dtype": args.amp_dtype,
                "grad_accum_steps": args.grad_accum_steps,
                "distill_weight": args.distill_weight,
                "direct_route_weight": args.direct_route_weight,
                "budget_weight": args.budget_weight,
                "log_every": args.log_every,
            },
            sort_keys=True,
        ),
        flush=True,
        file=sys.stderr,
    )
    history = train_steps(
        model,
        train_loader,
        steps=args.steps,
        optimizer=optimizer,
        device=device,
        teacher=teacher,
        distill_weight=args.distill_weight,
        direct_route_weight=args.direct_route_weight,
        budget_weight=args.budget_weight,
        lm_loss_weight=args.lm_loss_weight,
        answer_loss_weight=args.answer_loss_weight,
        amp=args.amp,
        amp_dtype=amp_dtype_from_arg(args.amp_dtype),
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        metrics_callback=emit_train_metrics,
    )

    eval_metrics = evaluate(
        model,
        eval_loader,
        device=device,
        batches=args.eval_batches,
        amp=args.amp,
        amp_dtype=amp_dtype_from_arg(args.amp_dtype),
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
