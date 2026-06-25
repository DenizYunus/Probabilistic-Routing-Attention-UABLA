# MemoryCache V1.4 Locked

Status: locked as the first MemoryCache profile family to clear the 80% KPI.

## Core Change

V1.4 changes leaf addressing, not just selection.

Earlier profiles summarized and symbolically sketched only the exact leaf payload.
That is too weak for records where the answer value sits near, but not inside,
the key/header bytes. V1.4 gives each leaf an address context:

```text
exact payload leaf:      small exact span revived into active cache
address context window:  neighboring bytes used for summary/sketch routing
```

The exact memory budget stays tied to the payload leaf. The context window only
improves addressing.

## Shared Settings

```bash
python scripts/run_memory_cache_ruler.py \
  --task ruler-aggregation \
  --seq-len 4096 \
  --needle-min-gap 3072 \
  --ruler-num-needles 8 \
  --ruler-aggregation-count 3 \
  --recent-window 256 \
  --summary-slots 1 \
  --compressor-type average \
  --symbolic-weight 8.0 \
  --symbolic-hash-bins 2048 \
  --symbolic-ngrams 2,3,4 \
  --source-target-mode dual \
  --source-coverage-weight 0.05 \
  --leaf-ce-weight 0.2 \
  --survival-loss-weight 1.0 \
  --query-window 96 \
  --address-context-left 32 \
  --address-context-right 4 \
  --symbolic-context-left 32 \
  --symbolic-context-right 4 \
  --neural-gate-loss-weight 0.25
```

## Efficient Profile

Keeps the original 320 active exact-token budget:

```bash
python scripts/run_memory_cache_ruler.py \
  --leaf-size 8 \
  --revived-top-k 8 \
  --symbolic-gate-top-k 16 \
  --neural-gate-top-k 16
```

Local MPS verification, 3000 steps, 128 eval batches:

```text
source_hit_rate:               89.48%
source_set_hit_rate:           60.17%
mean_active_exact_tokens:      320
active_token_compression:      12.8x
old_leaf_count:                480
neural_source_hit_rate:        79.39%
symbolic_source_hit_rate:      91.64%
```

## Quality Profile

Spends 384 active exact tokens, still under 10% of a 4096-byte context:

```bash
python scripts/run_memory_cache_ruler.py \
  --leaf-size 16 \
  --revived-top-k 8 \
  --symbolic-gate-top-k 16 \
  --neural-gate-top-k 16
```

Local MPS verification, 3000 steps, 128 eval batches:

```text
source_hit_rate:               94.99%
source_set_hit_rate:           74.72%
mean_active_exact_tokens:      384
active_token_compression:      10.67x
old_leaf_count:                240
neural_source_hit_rate:        88.21%
symbolic_source_hit_rate:      96.39%
```

## Interpretation

The original MemoryCache V1.3 gate-top8 profile reached 66.21% source-hit at
320 active exact tokens. V1.4 shows the missing piece was not only a sharper
selector. The cache needed richer address metadata per exact payload leaf.

This is LLM-relevant because real documents often separate identifiers,
headers, names, and values across nearby spans. A cache that keeps only the
payload as its address can prune the right value simply because its local bytes
do not resemble the query.

## Current KPI Status

```text
80% KPI: passed
90% KPI: efficient profile nearly reaches it; quality profile passes it
95% KPI: quality profile is effectively at the threshold and needs repeat seeds
```

## Next Checks

1. Repeat both profiles across seeds 1-3.
2. Test the same profiles on multi-needle and multi-hop RULER-lite.
3. Try 8192 context with the same active exact-token percentage.
4. Port the address-context idea into the main UABLA fusion path.
