# RULER Rerank V1 Locked

Status: locked as the first byte-LM RULER run to break the exact-match wall.

Date locked: 2026-06-30

Last updated: 2026-07-01 with rerank calibration sweep and repeat-seed validation.

## Locked Result

Benchmark:

```text
task: ruler-multineedle
input mode: token
sequence length: 4096
needle min gap: 3072
needles: 8
key length: 6
answer length: 12
aggregation count: 3
batch size: 8
```

Best checkpoint:

```text
token_ruler_kpi_rerank_tune_4096_03_sharp_s25_t075_best.pt
```

Source archive:

```text
token_ruler_kpi_rerank_tune_4096_outputs.zip
```

Final evaluation from the winning run:

```text
answer_accuracy:      92.7612%
answer_exact_match:   58.7891%
answer_loss:          0.54122
source_token_recall:  80.2083%
record_token_recall:  91.6667%
peak_memory_mb:       ~10166.37
```

Repeat-seed validation, eval-only over seeds 0-4:

```text
answer_accuracy mean:     92.7726%
answer_accuracy min/max:  92.7612% / 92.7775%
answer_exact_match mean:  58.8184%
answer_exact_match min/max: 58.7891% / 58.8379%
answer_loss mean:         0.54072
answer_loss min/max:      0.54050 / 0.54122
eval peak_memory_mb:      ~5.18-5.30GB
```

Harder-task eval-only validation:

```text
multi-needle, 8 records:   accuracy 92.76%, exact 58.79%
multi-needle, 16 records:  accuracy 85.12%, exact 34.86%
multi-needle, 24 records:  accuracy 81.60%, exact 25.20%
multi-hop, 8 chains:       accuracy 79.36%, exact 17.29%
multi-hop, 12 chains:      accuracy 70.61%, exact  6.35%
aggregation, 8 x 3 parts:  accuracy 51.74%, exact  0.00%
aggregation, 12 x 3 parts: accuracy 48.49%, exact  0.00%
```

Interpretation of harder validation:

```text
multi-needle degradation is mostly source-survival pressure
multi-hop needs explicit two-step lookup training
aggregation needs multi-span composition/order training, not just rerank
```

Previous best before rerank:

```text
answer_accuracy:      ~80.59%
answer_exact_match:   ~1.46%
peak exact_match:      3.125%
```

## Locked Recipe

The locked calibrated rerank profile is:

```text
--copy-logit-mode rerank
--eval-copy-logit-mode rerank
--copy-logit-scale 3.5
--eval-copy-logit-scale 2.5
--copy-logit-temperature 0.75
```

The routing profile remains the wide blend family:

```text
--hier-layer-thresholds 0.20,0.20,0.20,0.20
--hier-leaf-top-k 32
--hier-leaf-min-k 8
--hier-attention-chunk-size 64
```

Training auxiliaries:

```text
--answer-exactness-weight 1.0
--answer-exactness-temperature 0.10
--answer-loss-weight 5.0
--hier-route-weight 0.01
--hier-coverage-weight 0.03
--hier-record-route-weight 0.02
--copy-pointer-weight 0.01
```

## Interpretation

The routed candidate set is now good enough that the main failure is not source
survival. The main failure is coherent answer decoding.

Pure rerank treats every routed candidate start as a full answer proposal,
scores the proposed string with the model's answer-position logits, then copies
from the best proposal distribution. This is materially better than per-token
copy and better than mixing rerank with token-copy.

Initial rerank ordering:

```text
rerank          > blend-rerank sharp > blend-rerank > blend-seq
```

Calibration sweep ordering:

```text
scale 3.5 / eval scale 2.5 / temp 0.75  >  scale 3.0 / eval scale 2.0 / temp 0.75
> scale 3.0 / eval scale 2.0 / temp 0.90 > soft rerank
```

This means rerank should be treated as a first-class inference path, not just an
auxiliary training trick.

## Next Experiments

Next steps:

```text
1. train/evaluate the rerank path on 16-needle multi-needle
2. train a multi-hop-specific continuation
3. train an aggregation-specific composition continuation
4. test 8192 only after these harder 4096 tasks improve
```

The current profile is seed-validated at 4096 tokens. Do not widen routing again
until harder-task validation shows source survival is the limiter.
