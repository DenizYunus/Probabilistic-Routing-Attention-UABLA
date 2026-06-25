# MemoryCache Dual Router V1.2 Locked

Status: locked as the best current MemoryCache simulator design.

## Locked Quality Profile

```text
compressor_type = average
summary_slots = 1
slot_score_reduction = max
recent_window = 256
leaf_size = 16
revived_top_k = 4
symbolic_weight = 8.0
symbolic_ngrams = 2,3,4
symbolic_ngram_weights = unset
symbolic_hash_bins = 2048
revived_min_k = unset
dynamic_score_margin = 0.0
```

## Locked Efficient Profile

```text
compressor_type = average
summary_slots = 1
slot_score_reduction = max
recent_window = 256
leaf_size = 16
revived_top_k = 3
symbolic_weight = 8.0
symbolic_ngrams = 2,3,4
symbolic_ngram_weights = 0.25,1.0,0.5
symbolic_hash_bins = 2048
revived_min_k = unset
dynamic_score_margin = 0.0
```

## What Is Locked

Dual Router V1.2 combines two leaf-addressing signals:

```text
neural/average route score
+ multi-ngram symbolic byte-sketch score
```

The average summary remains the strongest stable neural address. The symbolic
path adds a cheap exact-ish retrieval signal for byte patterns such as keys,
codes, names, numbers, repeated identifiers, and short values.

The locked quality profile keeps fixed top-4 revival. This preserves the same
active exact-token budget as the previous average-summary baseline while
improving source-hit recall.

The locked efficient profile uses fixed top-3 revival with weighted
multi-ngram sketches. It spends fewer active exact tokens than the baseline
while still beating the previous top-4 single-ngram symbolic settings.

## Result That Locked It

RULER-lite byte multi-needle, 4096-byte context, 16 needles, recent window 256:

```text
baseline average top-4:        57.12% source-hit, 320 active tokens, 12.80x compression
w4/ng3/h2048 top-4:            65.29% source-hit, 320 active tokens, 12.80x compression
w6/ng3/h2048 top-4:            67.91% source-hit, 320 active tokens, 12.80x compression
w8/ng3/h2048 top-4:            69.34% source-hit, 320 active tokens, 12.80x compression
w8/ng3/h2048 top-3:            65.64% source-hit, 304 active tokens, 13.47x compression
w8/ng3/h2048 top-2:            58.63% source-hit, 288 active tokens, 14.22x compression
equal ng2,3,4 top-4:           71.38% source-hit, 320 active tokens, 12.80x compression
weighted ng2,3,4 top-3:        67.63% source-hit, 304 active tokens, 13.47x compression
```

The locked quality profile improves source-hit recall by +12.22 points over
the average baseline with no increase in active exact KV tokens. The V1.2
multi-ngram quality profile improves that to +14.26 points over baseline.

The locked efficient profile improves source-hit recall by +10.51 points while
reducing active exact tokens from 320 to 304.

Diagnostics from the multi-ngram sweep:

```text
equal ng2,3,4 top-4:
  combined source rank 5.01
  neural source rank   7.09
  symbolic source rank 90.80
  neural top-k hit     51.80%
  symbolic top-k hit    9.31%

weighted ng2,3,4 top-3:
  combined source rank 5.24
  neural source rank   7.53
  symbolic source rank 78.80
  neural top-k hit     44.38%
  symbolic top-k hit    9.39%
```

The symbolic path is not strong enough alone. It is locked as a cheap reranking
bias that improves the combined route.

## What Is Not Locked

Dynamic revival remains experimental:

```text
revived_min_k
dynamic_score_margin
```

The first useful dynamic setting (`revived_min_k=2`, `margin=1.0`) slightly
reduced active tokens but did not beat fixed top-4 symbolic routing:

```text
min2 margin 1.0 symbolic 0.5: 56.78% source-hit, 313.8 active tokens, 13.05x compression
```

Dynamic routing should be treated as a secondary cost knob, not the primary
quality path for v1.2. Fixed top-k profiles are the locked path.

Aggregation-specific answer-set supervision is experimental, not locked. The
plain aggregation run was weak:

```text
weighted ng2,3,4 top-4 on aggregation: 34.08% source-hit, 320 active tokens, 12.80x compression
```

The likely issue is target mismatch: aggregation answers depend on several
source leaves, but per-token supervision rewards one source byte at a time.
The runner now exposes:

```bash
--source-target-mode answer-set
```

In that mode, each answer query is trained and evaluated against the full set
of answer source leaves. The logged `source_hit_rate` becomes source-leaf
recall for that set.

The experimental coverage objective:

```bash
--source-coverage-weight 0.2
--source-coverage-temperature 1.0
```

adds a uniform-target loss over all source leaves. This is designed to prevent
answer-set training from collapsing onto only one relevant source leaf.

## Locked Aggregation-Friendly Profile

Dual-target aggregation is locked as the current best profile for aggregation
without damaging the main per-token route:

```text
compressor_type = average
summary_slots = 1
slot_score_reduction = max
recent_window = 256
leaf_size = 16
revived_top_k = 4
symbolic_weight = 8.0
symbolic_ngrams = 2,3,4
symbolic_ngram_weights = unset
symbolic_hash_bins = 2048
source_target_mode = dual
source_coverage_weight = 0.05
source_coverage_temperature = 1.0
```

RULER-lite byte aggregation, 4096-byte context, 8 records, 3-part answer:

```text
per-token anchor:       57.95% source-hit, 320 active tokens, 12.80x compression
answer-set cov 0.20:    41.80% set-hit,    320 active tokens, 12.80x compression
dual cov 0.05:          57.82% source-hit, 41.49% set-hit, 320 active tokens, 12.80x compression
dual cov 0.10:          57.76% source-hit, 41.55% set-hit, 320 active tokens, 12.80x compression
dual cov 0.20:          57.25% source-hit, 41.86% set-hit, 320 active tokens, 12.80x compression
dual cov 0.50:          55.98% source-hit, 42.43% set-hit, 320 active tokens, 12.80x compression
```

Dual `0.05` is locked because it adds meaningful source-set coverage while
only dropping per-token source-hit by 0.14 points from the anchor.

## Experimental Diverse Revival

Plain top-k revival can select redundant leaves. Diverse revival is available
as an experimental selector:

```text
revival_selection = diverse
diversity_weight = sweep value
diversity_candidate_multiplier = 4
```

It keeps the active exact-token budget unchanged. The first selected leaf is
the highest-scoring leaf; later leaves are selected from a high-score candidate
pool with a cosine-similarity penalty against already selected leaf summaries.
This is designed for multi-source tasks such as aggregation and multi-hop.

## Why This Matters

This is not a larger neural module. It is a memory-indexing improvement:

```text
same or lower active token budget
same average summary cache
cheap symbolic sketch added to routing
better long-context source survival
```

The next experiments should test whether the locked profile transfers beyond
the current multi-needle setup before adding larger learned summaries.

## Next Steps

Recommended follow-up experiments:

```text
1. Validate dual cov 0.05 on multi-hop and harder aggregation variants.
2. Increase task hardness: more needles, longer key/value codes, larger gaps.
3. Test longer contexts: 8192 bytes, then 16384 bytes.
4. Add source-hit diagnostics by answer position and source depth.
5. Later, fuse MemoryCache routing with UABLA attention routing.
```
