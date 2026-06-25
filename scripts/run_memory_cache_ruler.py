#!/usr/bin/env python3
"""Train/evaluate MemoryCache source-hit survival on RULER-lite byte tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from uabla.byte_lm import (
    BYTE_LM_VOCAB_SIZE,
    DEFAULT_BYTE_CORPUS,
    bytes_to_ids,
    ensure_min_bytes,
    make_byte_loader,
    split_byte_stream,
)
from uabla.experiment import move_batch
from uabla.memory_cache import (
    MemoryCacheProbeMetrics,
    MemoryCacheRouteProbe,
    RevivalCacheConfig,
    collect_probe_metrics,
    collect_probe_set_metrics,
    memory_cache_source_leaf_loss,
    memory_cache_source_set_coverage_loss,
    memory_cache_source_set_leaf_loss,
    memory_cache_source_set_survival_loss,
    memory_cache_revived_leaf_diversity_loss,
    memory_cache_summary_diversity_loss,
    memory_cache_topk_survival_loss,
)
from uabla.ruler_lite import (
    ByteRulerLiteConfig,
    ByteRulerLiteDataset,
    RULER_LITE_CLI_TO_TASK,
)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_float_csv(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one number")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(RULER_LITE_CLI_TO_TASK), default="ruler-multineedle")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--train-size", type=int, default=200000)
    parser.add_argument("--eval-size", type=int, default=4096)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--recent-window", type=int, default=512)
    parser.add_argument("--summary-slots", type=int, default=4)
    parser.add_argument("--address-context-left", type=int, default=0)
    parser.add_argument("--address-context-right", type=int, default=0)
    parser.add_argument("--revived-top-k", type=int, default=8)
    parser.add_argument("--revived-min-k", type=int)
    parser.add_argument("--dynamic-score-margin", type=float, default=0.0)
    parser.add_argument("--slot-score-reduction", choices=["max", "logsumexp", "mean"], default="max")
    parser.add_argument("--slot-score-temperature", type=float, default=1.0)
    parser.add_argument("--symbolic-weight", type=float, default=0.0)
    parser.add_argument("--symbolic-hash-bins", type=int, default=512)
    parser.add_argument("--symbolic-ngram", type=int, default=3)
    parser.add_argument("--symbolic-ngrams", type=parse_int_csv)
    parser.add_argument("--symbolic-ngram-weights", type=parse_float_csv)
    parser.add_argument("--symbolic-gate-top-k", type=int)
    parser.add_argument("--symbolic-context-left", type=int, default=0)
    parser.add_argument("--symbolic-context-right", type=int, default=0)
    parser.add_argument("--symbolic-score-norm", choices=["none", "zscore", "minmax"], default="none")
    parser.add_argument(
        "--revival-selection",
        choices=["topk", "diverse", "channel-mix", "rank-fusion", "gated-rerank"],
        default="topk",
    )
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--diversity-candidate-multiplier", type=int, default=4)
    parser.add_argument("--gated-rerank-top-k", type=int)
    parser.add_argument("--channel-mix-neural-k", type=int, default=1)
    parser.add_argument("--channel-mix-symbolic-k", type=int, default=1)
    parser.add_argument("--rank-fusion-neural-weight", type=float, default=1.0)
    parser.add_argument("--rank-fusion-symbolic-weight", type=float, default=1.0)
    parser.add_argument("--rank-fusion-combined-weight", type=float, default=1.0)
    parser.add_argument("--rank-fusion-offset", type=float, default=8.0)
    parser.add_argument("--adaptive-budget-mode", choices=["none", "entropy"], default="none")
    parser.add_argument("--adaptive-min-k", type=int)
    parser.add_argument("--adaptive-entropy-threshold", type=float, default=0.75)
    parser.add_argument("--query-window", type=int, default=96)
    parser.add_argument("--query-pooling", choices=["mean", "last", "attn", "mean-pos"], default="mean")
    parser.add_argument("--query-index-embedding", action="store_true")
    parser.add_argument(
        "--compressor-type",
        choices=["multislot", "average", "hybrid"],
        default="multislot",
    )
    parser.add_argument("--needle-code-length", type=int, default=12)
    parser.add_argument("--needle-min-gap", type=int, default=3072)
    parser.add_argument("--ruler-num-needles", type=int, default=8)
    parser.add_argument("--ruler-key-length", type=int, default=6)
    parser.add_argument("--ruler-aggregation-count", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--leaf-ce-weight", type=float, default=0.2)
    parser.add_argument("--survival-loss-weight", type=float, default=1.0)
    parser.add_argument("--survival-margin", type=float, default=0.0)
    parser.add_argument("--survival-temperature", type=float, default=1.0)
    parser.add_argument("--budget-ladder-weight", type=float, default=0.0)
    parser.add_argument("--budget-ladder-top-ks", type=parse_int_csv)
    parser.add_argument("--neural-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--neural-gate-top-k", type=int)
    parser.add_argument("--summary-diversity-weight", type=float, default=0.0)
    parser.add_argument("--revived-diversity-weight", type=float, default=0.0)
    parser.add_argument("--source-coverage-weight", type=float, default=0.0)
    parser.add_argument("--source-coverage-temperature", type=float, default=1.0)
    parser.add_argument(
        "--source-target-mode",
        choices=["per-token", "answer-set", "dual"],
        default="per-token",
        help=(
            "per-token trains each answer byte toward its own source byte; "
            "answer-set trains each answer query toward all source leaves in the answer; "
            "dual keeps per-token routing losses and adds answer-set coverage as an auxiliary loss."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    raw_bytes = args.text_file.read_bytes() if args.text_file is not None else DEFAULT_BYTE_CORPUS
    min_bytes = max(args.seq_len * 2 + 2, 4096)
    byte_ids = bytes_to_ids(ensure_min_bytes(raw_bytes, min_bytes))
    train_bytes, eval_bytes = split_byte_stream(
        byte_ids,
        seq_len=args.seq_len,
        eval_fraction=args.eval_fraction,
    )
    ruler_task = RULER_LITE_CLI_TO_TASK[args.task]
    train_dataset = ByteRulerLiteDataset(
        train_bytes,
        ByteRulerLiteConfig(
            seq_len=args.seq_len,
            dataset_size=args.train_size,
            seed=args.seed,
            task=ruler_task,
            code_length=args.needle_code_length,
            key_length=args.ruler_key_length,
            num_needles=args.ruler_num_needles,
            aggregation_count=args.ruler_aggregation_count,
            min_gap=args.needle_min_gap,
        ),
    )
    eval_dataset = ByteRulerLiteDataset(
        eval_bytes,
        ByteRulerLiteConfig(
            seq_len=args.seq_len,
            dataset_size=args.eval_size,
            seed=args.seed + 10_000,
            task=ruler_task,
            code_length=args.needle_code_length,
            key_length=args.ruler_key_length,
            num_needles=args.ruler_num_needles,
            aggregation_count=args.ruler_aggregation_count,
            min_gap=args.needle_min_gap,
        ),
    )
    train_loader = make_byte_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = make_byte_loader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    cache_config = RevivalCacheConfig(
        leaf_size=args.leaf_size,
        recent_window=args.recent_window,
        summary_slots=args.summary_slots,
        address_context_left=args.address_context_left,
        address_context_right=args.address_context_right,
        revived_top_k=args.revived_top_k,
        revived_min_k=args.revived_min_k,
        dynamic_score_margin=args.dynamic_score_margin,
        slot_score_reduction=args.slot_score_reduction,
        slot_score_temperature=args.slot_score_temperature,
        symbolic_weight=args.symbolic_weight,
        symbolic_hash_bins=args.symbolic_hash_bins,
        symbolic_ngram=args.symbolic_ngram,
        symbolic_ngrams=args.symbolic_ngrams,
        symbolic_ngram_weights=args.symbolic_ngram_weights,
        symbolic_gate_top_k=args.symbolic_gate_top_k,
        symbolic_context_left=args.symbolic_context_left,
        symbolic_context_right=args.symbolic_context_right,
        symbolic_score_norm=args.symbolic_score_norm,
        revival_selection=args.revival_selection,
        diversity_weight=args.diversity_weight,
        diversity_candidate_multiplier=args.diversity_candidate_multiplier,
        gated_rerank_top_k=args.gated_rerank_top_k,
        channel_mix_neural_k=args.channel_mix_neural_k,
        channel_mix_symbolic_k=args.channel_mix_symbolic_k,
        rank_fusion_neural_weight=args.rank_fusion_neural_weight,
        rank_fusion_symbolic_weight=args.rank_fusion_symbolic_weight,
        rank_fusion_combined_weight=args.rank_fusion_combined_weight,
        rank_fusion_offset=args.rank_fusion_offset,
        adaptive_budget_mode=args.adaptive_budget_mode,
        adaptive_min_k=args.adaptive_min_k,
        adaptive_entropy_threshold=args.adaptive_entropy_threshold,
    )
    model = MemoryCacheRouteProbe(
        vocab_size=BYTE_LM_VOCAB_SIZE,
        hidden_size=args.hidden_size,
        max_seq_len=args.seq_len,
        cache_config=cache_config,
        query_window=args.query_window,
        compressor_type=args.compressor_type,
        query_pooling=args.query_pooling,
        use_query_index_embedding=args.query_index_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        json.dumps(
            {
                "phase": "start",
                "task": args.task,
                "ruler_task": ruler_task,
                "seq_len": args.seq_len,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden_size,
                "leaf_size": args.leaf_size,
                "recent_window": args.recent_window,
                "summary_slots": args.summary_slots,
                "address_context_left": args.address_context_left,
                "address_context_right": args.address_context_right,
                "revived_top_k": args.revived_top_k,
                "revived_min_k": args.revived_min_k,
                "dynamic_score_margin": args.dynamic_score_margin,
                "slot_score_reduction": args.slot_score_reduction,
                "slot_score_temperature": args.slot_score_temperature,
                "symbolic_weight": args.symbolic_weight,
                "symbolic_hash_bins": args.symbolic_hash_bins,
                "symbolic_ngram": args.symbolic_ngram,
                "symbolic_ngrams": args.symbolic_ngrams,
                "symbolic_ngram_weights": args.symbolic_ngram_weights,
                "symbolic_gate_top_k": args.symbolic_gate_top_k,
                "symbolic_context_left": args.symbolic_context_left,
                "symbolic_context_right": args.symbolic_context_right,
                "symbolic_score_norm": args.symbolic_score_norm,
                "revival_selection": args.revival_selection,
                "diversity_weight": args.diversity_weight,
                "diversity_candidate_multiplier": args.diversity_candidate_multiplier,
                "gated_rerank_top_k": args.gated_rerank_top_k,
                "channel_mix_neural_k": args.channel_mix_neural_k,
                "channel_mix_symbolic_k": args.channel_mix_symbolic_k,
                "rank_fusion_neural_weight": args.rank_fusion_neural_weight,
                "rank_fusion_symbolic_weight": args.rank_fusion_symbolic_weight,
                "rank_fusion_combined_weight": args.rank_fusion_combined_weight,
                "rank_fusion_offset": args.rank_fusion_offset,
                "adaptive_budget_mode": args.adaptive_budget_mode,
                "adaptive_min_k": args.adaptive_min_k,
                "adaptive_entropy_threshold": args.adaptive_entropy_threshold,
                "query_window": args.query_window,
                "query_pooling": args.query_pooling,
                "query_index_embedding": args.query_index_embedding,
                "compressor_type": args.compressor_type,
                "needle_code_length": args.needle_code_length,
                "needle_min_gap": args.needle_min_gap,
                "ruler_num_needles": args.ruler_num_needles,
                "ruler_key_length": args.ruler_key_length,
                "ruler_aggregation_count": args.ruler_aggregation_count,
                "leaf_ce_weight": args.leaf_ce_weight,
                "survival_loss_weight": args.survival_loss_weight,
                "survival_margin": args.survival_margin,
                "survival_temperature": args.survival_temperature,
                "budget_ladder_weight": args.budget_ladder_weight,
                "budget_ladder_top_ks": args.budget_ladder_top_ks,
                "neural_gate_loss_weight": args.neural_gate_loss_weight,
                "neural_gate_top_k": args.neural_gate_top_k,
                "summary_diversity_weight": args.summary_diversity_weight,
                "revived_diversity_weight": args.revived_diversity_weight,
                "source_coverage_weight": args.source_coverage_weight,
                "source_coverage_temperature": args.source_coverage_temperature,
                "source_target_mode": args.source_target_mode,
                "device": str(device),
                "train_stream_bytes": int(train_bytes.numel()),
                "eval_stream_bytes": int(eval_bytes.numel()),
                "raw_corpus_bytes": len(raw_bytes),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model.train()
    stream = iter(train_loader)
    start_time = time.perf_counter()
    token_count = 0
    for step in range(1, args.steps + 1):
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(train_loader)
            batch = next(stream)
        batch = move_batch(batch, device)
        output = model(batch["input_ids"], batch["answer_index"])
        source_indices = source_indices_for_mode(batch, args.source_target_mode)
        ce_loss, survival_loss, coverage_loss = source_losses(output, source_indices, args)
        ladder_loss = budget_ladder_loss(output, source_indices, args)
        gate_loss = neural_gate_loss(output, source_indices, args)
        diversity_loss = memory_cache_summary_diversity_loss(output)
        revived_diversity_loss = memory_cache_revived_leaf_diversity_loss(output)
        loss = (
            args.leaf_ce_weight * ce_loss
            + args.survival_loss_weight * survival_loss
            + args.source_coverage_weight * coverage_loss
            + args.budget_ladder_weight * ladder_loss
            + args.neural_gate_loss_weight * gate_loss
            + args.summary_diversity_weight * diversity_loss
            + args.revived_diversity_weight * revived_diversity_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        token_count += batch["input_ids"].numel()

        if step % args.log_every == 0 or step == args.steps:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            metrics = collect_metrics_for_mode(output, source_indices, args)
            print(
                json.dumps(
                    {
                        "phase": "train",
                        "step": step,
                        "steps": args.steps,
                        "progress_pct": 100.0 * step / args.steps,
                        "elapsed_sec": elapsed,
                        "eta_sec": elapsed * (args.steps - step) / max(step, 1),
                        "tokens_per_second": token_count / elapsed,
                        "loss": float(loss.detach().item()),
                        "ce_loss": float(ce_loss.detach().item()),
                        "survival_loss": float(survival_loss.detach().item()),
                        "coverage_loss": float(coverage_loss.detach().item()),
                        "budget_ladder_loss": float(ladder_loss.detach().item()),
                        "neural_gate_loss": float(gate_loss.detach().item()),
                        "summary_diversity_loss": float(diversity_loss.detach().item()),
                        "revived_diversity_loss": float(revived_diversity_loss.detach().item()),
                        **metrics_to_json(metrics),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    eval_metrics = evaluate(model, eval_loader, args=args, device=device)
    print(
        json.dumps(
            {
                "phase": "eval",
                **metrics_to_json(eval_metrics),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "eval": metrics_to_json(eval_metrics),
            },
            args.save_checkpoint,
        )


def evaluate(
    model: MemoryCacheRouteProbe,
    loader,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> MemoryCacheProbeMetrics:
    model.eval()
    totals: dict[str, float] = {}
    optional_counts: dict[str, int] = {}
    count = 0
    max_active = 0
    old_leaf_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.eval_batches:
                break
            batch = move_batch(batch, device)
            output = model(batch["input_ids"], batch["answer_index"])
            source_indices = source_indices_for_mode(batch, args.source_target_mode)
            metrics = collect_metrics_for_mode(output, source_indices, args)
            values = metrics_to_json(metrics)
            for key, value in values.items():
                if key in {"max_active_exact_tokens", "old_leaf_count"}:
                    continue
                if value is None:
                    continue
                totals[key] = totals.get(key, 0.0) + float(value)
                optional_counts[key] = optional_counts.get(key, 0) + 1
            max_active = max(max_active, metrics.max_active_exact_tokens)
            old_leaf_count = metrics.old_leaf_count
            count += 1
    count = max(count, 1)
    mean_active = totals["mean_active_exact_tokens"] / count
    optional_metrics = {
        key: totals[key] / optional_counts[key]
        for key in (
            "source_coverage_loss",
            "source_set_hit_rate",
            "source_set_leaf_accuracy",
            "revived_diversity_loss",
            "combined_source_rank",
            "neural_source_rank",
            "symbolic_source_rank",
            "neural_source_hit_rate",
            "symbolic_source_hit_rate",
        )
        if key in totals and optional_counts.get(key, 0) > 0
    }
    return MemoryCacheProbeMetrics(
        source_hit_rate=totals["source_hit_rate"] / count,
        source_leaf_accuracy=totals["source_leaf_accuracy"] / count,
        source_leaf_loss=totals["source_leaf_loss"] / count,
        source_survival_loss=totals["source_survival_loss"] / count,
        mean_active_exact_tokens=mean_active,
        max_active_exact_tokens=max_active,
        old_leaf_count=old_leaf_count,
        active_token_compression_ratio=args.seq_len / max(mean_active, 1.0),
        **optional_metrics,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.train_size <= 0 or args.eval_size <= 0:
        raise ValueError("--train-size and --eval-size must be positive")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")
    if args.address_context_left < 0 or args.address_context_right < 0:
        raise ValueError("--address-context-left/right cannot be negative")
    if args.revived_min_k is not None:
        if args.revived_min_k <= 0:
            raise ValueError("--revived-min-k must be positive when provided")
        if args.revived_min_k > args.revived_top_k:
            raise ValueError("--revived-min-k cannot exceed --revived-top-k")
    if args.dynamic_score_margin < 0:
        raise ValueError("--dynamic-score-margin cannot be negative")
    if args.query_window <= 0:
        raise ValueError("--query-window must be positive")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if args.leaf_ce_weight < 0:
        raise ValueError("--leaf-ce-weight cannot be negative")
    if args.survival_loss_weight < 0:
        raise ValueError("--survival-loss-weight cannot be negative")
    if args.leaf_ce_weight == 0 and args.survival_loss_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    if args.survival_temperature <= 0:
        raise ValueError("--survival-temperature must be positive")
    if args.slot_score_temperature <= 0:
        raise ValueError("--slot-score-temperature must be positive")
    if args.symbolic_weight < 0:
        raise ValueError("--symbolic-weight cannot be negative")
    if args.symbolic_hash_bins <= 0:
        raise ValueError("--symbolic-hash-bins must be positive")
    if args.symbolic_ngram <= 0:
        raise ValueError("--symbolic-ngram must be positive")
    if args.symbolic_ngrams is not None:
        if any(value <= 0 for value in args.symbolic_ngrams):
            raise ValueError("--symbolic-ngrams values must be positive")
    if args.symbolic_ngram_weights is not None:
        if args.symbolic_ngrams is None:
            raise ValueError("--symbolic-ngram-weights requires --symbolic-ngrams")
        if len(args.symbolic_ngram_weights) != len(args.symbolic_ngrams):
            raise ValueError("--symbolic-ngram-weights length must match --symbolic-ngrams")
        if any(value < 0 for value in args.symbolic_ngram_weights):
            raise ValueError("--symbolic-ngram-weights cannot contain negative values")
        if sum(args.symbolic_ngram_weights) <= 0:
            raise ValueError("--symbolic-ngram-weights must have positive sum")
    if args.symbolic_gate_top_k is not None and args.symbolic_gate_top_k <= 0:
        raise ValueError("--symbolic-gate-top-k must be positive when provided")
    if args.symbolic_context_left < 0 or args.symbolic_context_right < 0:
        raise ValueError("--symbolic-context-left/right cannot be negative")
    if args.gated_rerank_top_k is not None and args.gated_rerank_top_k <= 0:
        raise ValueError("--gated-rerank-top-k must be positive when provided")
    if args.budget_ladder_weight < 0:
        raise ValueError("--budget-ladder-weight cannot be negative")
    if args.budget_ladder_weight > 0 and args.budget_ladder_top_ks is None:
        raise ValueError("--budget-ladder-weight requires --budget-ladder-top-ks")
    if args.budget_ladder_top_ks is not None:
        if any(value <= 0 for value in args.budget_ladder_top_ks):
            raise ValueError("--budget-ladder-top-ks values must be positive")
        if max(args.budget_ladder_top_ks) > args.revived_top_k:
            raise ValueError("--budget-ladder-top-ks cannot exceed --revived-top-k")
    if args.neural_gate_loss_weight < 0:
        raise ValueError("--neural-gate-loss-weight cannot be negative")
    if args.neural_gate_loss_weight > 0 and args.neural_gate_top_k is None and args.symbolic_gate_top_k is None:
        raise ValueError("--neural-gate-loss-weight requires --neural-gate-top-k or --symbolic-gate-top-k")
    if args.neural_gate_top_k is not None and args.neural_gate_top_k <= 0:
        raise ValueError("--neural-gate-top-k must be positive when provided")
    if args.summary_diversity_weight < 0:
        raise ValueError("--summary-diversity-weight cannot be negative")
    if args.revived_diversity_weight < 0:
        raise ValueError("--revived-diversity-weight cannot be negative")
    if args.diversity_weight < 0:
        raise ValueError("--diversity-weight cannot be negative")
    if args.diversity_candidate_multiplier <= 0:
        raise ValueError("--diversity-candidate-multiplier must be positive")
    if args.diversity_weight > 0 and args.revival_selection != "diverse":
        raise ValueError("--diversity-weight requires --revival-selection diverse")
    if args.channel_mix_neural_k < 0:
        raise ValueError("--channel-mix-neural-k cannot be negative")
    if args.channel_mix_symbolic_k < 0:
        raise ValueError("--channel-mix-symbolic-k cannot be negative")
    if args.revival_selection == "channel-mix" and args.channel_mix_neural_k == 0 and args.channel_mix_symbolic_k == 0:
        raise ValueError("channel-mix requires at least one channel-specific slot")
    if args.rank_fusion_neural_weight < 0:
        raise ValueError("--rank-fusion-neural-weight cannot be negative")
    if args.rank_fusion_symbolic_weight < 0:
        raise ValueError("--rank-fusion-symbolic-weight cannot be negative")
    if args.rank_fusion_combined_weight < 0:
        raise ValueError("--rank-fusion-combined-weight cannot be negative")
    if (
        args.rank_fusion_neural_weight
        + args.rank_fusion_symbolic_weight
        + args.rank_fusion_combined_weight
        <= 0
    ):
        raise ValueError("at least one rank-fusion weight must be positive")
    if args.rank_fusion_offset <= 0:
        raise ValueError("--rank-fusion-offset must be positive")
    if args.adaptive_min_k is not None:
        if args.adaptive_min_k <= 0:
            raise ValueError("--adaptive-min-k must be positive when provided")
        if args.adaptive_min_k > args.revived_top_k:
            raise ValueError("--adaptive-min-k cannot exceed --revived-top-k")
    if not 0.0 <= args.adaptive_entropy_threshold <= 1.0:
        raise ValueError("--adaptive-entropy-threshold must be in [0, 1]")
    if args.adaptive_budget_mode == "entropy" and args.adaptive_min_k is None:
        raise ValueError("--adaptive-budget-mode entropy requires --adaptive-min-k")
    if args.source_coverage_weight < 0:
        raise ValueError("--source-coverage-weight cannot be negative")
    if args.source_coverage_temperature <= 0:
        raise ValueError("--source-coverage-temperature must be positive")
    if args.source_target_mode in {"answer-set", "dual"} and args.task != "ruler-aggregation":
        raise ValueError("--source-target-mode answer-set/dual is intended for --task ruler-aggregation")
    if args.source_coverage_weight > 0 and args.source_target_mode not in {"answer-set", "dual"}:
        raise ValueError("--source-coverage-weight requires --source-target-mode answer-set or dual")
    if args.ruler_num_needles <= 0:
        raise ValueError("--ruler-num-needles must be positive")
    if args.ruler_key_length <= 0:
        raise ValueError("--ruler-key-length must be positive")
    if args.ruler_aggregation_count <= 0:
        raise ValueError("--ruler-aggregation-count must be positive")
    if (
        args.task == "ruler-aggregation"
        and args.ruler_num_needles < args.ruler_aggregation_count
    ):
        raise ValueError("--ruler-num-needles must be >= --ruler-aggregation-count")


def source_indices_for_mode(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    source_indices = batch["answer_source_index"]
    if mode in {"per-token", "dual"}:
        return source_indices
    if mode == "answer-set":
        return answer_source_set_indices(source_indices, query_count=batch["answer_index"].shape[1])
    raise ValueError(f"unknown source target mode: {mode}")


def answer_source_set_indices(
    source_indices: torch.Tensor,
    *,
    query_count: int | None = None,
) -> torch.Tensor:
    if source_indices.ndim == 3:
        return source_indices
    if source_indices.ndim != 2:
        raise ValueError("source_indices must have shape [batch, queries] or [batch, queries, sources]")
    if query_count is None:
        query_count = source_indices.shape[1]
    return source_indices.unsqueeze(1).expand(-1, query_count, -1)


def source_losses(
    output,
    source_indices: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.source_target_mode == "answer-set":
        ce_loss = memory_cache_source_set_leaf_loss(
            output,
            source_indices,
            leaf_size=args.leaf_size,
        )
        survival_loss = memory_cache_source_set_survival_loss(
            output,
            source_indices,
            leaf_size=args.leaf_size,
            top_k=args.revived_top_k,
            margin=args.survival_margin,
            temperature=args.survival_temperature,
        )
        coverage_loss = memory_cache_source_set_coverage_loss(
            output,
            source_indices,
            leaf_size=args.leaf_size,
            temperature=args.source_coverage_temperature,
        )
        return ce_loss, survival_loss, coverage_loss
    if args.source_target_mode == "dual":
        ce_loss = memory_cache_source_leaf_loss(
            output,
            source_indices,
            leaf_size=args.leaf_size,
        )
        survival_loss = memory_cache_topk_survival_loss(
            output,
            source_indices,
            leaf_size=args.leaf_size,
            top_k=args.revived_top_k,
            margin=args.survival_margin,
            temperature=args.survival_temperature,
        )
        coverage_loss = memory_cache_source_set_coverage_loss(
            output,
            answer_source_set_indices(source_indices),
            leaf_size=args.leaf_size,
            temperature=args.source_coverage_temperature,
        )
        return ce_loss, survival_loss, coverage_loss
    ce_loss = memory_cache_source_leaf_loss(
        output,
        source_indices,
        leaf_size=args.leaf_size,
    )
    survival_loss = memory_cache_topk_survival_loss(
        output,
        source_indices,
        leaf_size=args.leaf_size,
        top_k=args.revived_top_k,
        margin=args.survival_margin,
        temperature=args.survival_temperature,
    )
    coverage_loss = output.leaf_scores.sum() * 0.0
    return ce_loss, survival_loss, coverage_loss


def budget_ladder_loss(
    output,
    source_indices: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.budget_ladder_weight == 0 or args.budget_ladder_top_ks is None:
        return output.leaf_scores.sum() * 0.0
    losses = []
    for top_k in args.budget_ladder_top_ks:
        if args.source_target_mode == "answer-set":
            losses.append(
                memory_cache_source_set_survival_loss(
                    output,
                    source_indices,
                    leaf_size=args.leaf_size,
                    top_k=top_k,
                    margin=args.survival_margin,
                    temperature=args.survival_temperature,
                )
            )
        else:
            losses.append(
                memory_cache_topk_survival_loss(
                    output,
                    source_indices,
                    leaf_size=args.leaf_size,
                    top_k=top_k,
                    margin=args.survival_margin,
                    temperature=args.survival_temperature,
                )
            )
    return torch.stack(losses).mean()


def neural_gate_loss(
    output,
    source_indices: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.neural_gate_loss_weight == 0:
        return output.leaf_scores.sum() * 0.0
    if output.neural_leaf_scores is None:
        return output.leaf_scores.sum() * 0.0
    top_k = args.neural_gate_top_k or args.symbolic_gate_top_k or args.revived_top_k
    neural_output = replace(output, leaf_scores=output.neural_leaf_scores)
    if args.source_target_mode == "answer-set":
        return memory_cache_source_set_survival_loss(
            neural_output,
            source_indices,
            leaf_size=args.leaf_size,
            top_k=top_k,
            margin=args.survival_margin,
            temperature=args.survival_temperature,
        )
    return memory_cache_topk_survival_loss(
        neural_output,
        source_indices,
        leaf_size=args.leaf_size,
        top_k=top_k,
        margin=args.survival_margin,
        temperature=args.survival_temperature,
    )


def collect_metrics_for_mode(
    output,
    source_indices: torch.Tensor,
    args: argparse.Namespace,
) -> MemoryCacheProbeMetrics:
    if args.source_target_mode == "answer-set":
        return collect_probe_set_metrics(
            output,
            source_indices,
            seq_len=args.seq_len,
            leaf_size=args.leaf_size,
        )
    if args.source_target_mode == "dual":
        per_token_metrics = collect_probe_metrics(
            output,
            source_indices,
            seq_len=args.seq_len,
            leaf_size=args.leaf_size,
        )
        set_metrics = collect_probe_set_metrics(
            output,
            answer_source_set_indices(source_indices),
            seq_len=args.seq_len,
            leaf_size=args.leaf_size,
        )
        return MemoryCacheProbeMetrics(
            source_hit_rate=per_token_metrics.source_hit_rate,
            source_leaf_accuracy=per_token_metrics.source_leaf_accuracy,
            source_leaf_loss=per_token_metrics.source_leaf_loss,
            source_survival_loss=per_token_metrics.source_survival_loss,
            mean_active_exact_tokens=per_token_metrics.mean_active_exact_tokens,
            max_active_exact_tokens=per_token_metrics.max_active_exact_tokens,
            old_leaf_count=per_token_metrics.old_leaf_count,
            active_token_compression_ratio=per_token_metrics.active_token_compression_ratio,
            source_coverage_loss=set_metrics.source_coverage_loss,
            source_set_hit_rate=set_metrics.source_hit_rate,
            source_set_leaf_accuracy=set_metrics.source_leaf_accuracy,
            revived_diversity_loss=per_token_metrics.revived_diversity_loss,
            combined_source_rank=per_token_metrics.combined_source_rank,
            neural_source_rank=per_token_metrics.neural_source_rank,
            symbolic_source_rank=per_token_metrics.symbolic_source_rank,
            neural_source_hit_rate=per_token_metrics.neural_source_hit_rate,
            symbolic_source_hit_rate=per_token_metrics.symbolic_source_hit_rate,
        )
    return collect_probe_metrics(
        output,
        source_indices,
        seq_len=args.seq_len,
        leaf_size=args.leaf_size,
    )


def metrics_to_json(metrics: MemoryCacheProbeMetrics) -> dict[str, Any]:
    values = {
        "source_hit_rate": metrics.source_hit_rate,
        "source_leaf_accuracy": metrics.source_leaf_accuracy,
        "source_leaf_loss": metrics.source_leaf_loss,
        "source_survival_loss": metrics.source_survival_loss,
        "mean_active_exact_tokens": metrics.mean_active_exact_tokens,
        "max_active_exact_tokens": metrics.max_active_exact_tokens,
        "old_leaf_count": metrics.old_leaf_count,
        "active_token_compression_ratio": metrics.active_token_compression_ratio,
    }
    optional_values = {
        "source_coverage_loss": metrics.source_coverage_loss,
        "source_set_hit_rate": metrics.source_set_hit_rate,
        "source_set_leaf_accuracy": metrics.source_set_leaf_accuracy,
        "revived_diversity_loss": metrics.revived_diversity_loss,
        "combined_source_rank": metrics.combined_source_rank,
        "neural_source_rank": metrics.neural_source_rank,
        "symbolic_source_rank": metrics.symbolic_source_rank,
        "neural_source_hit_rate": metrics.neural_source_hit_rate,
        "symbolic_source_hit_rate": metrics.symbolic_source_hit_rate,
    }
    values.update({key: value for key, value in optional_values.items() if value is not None})
    return values


if __name__ == "__main__":
    main()
