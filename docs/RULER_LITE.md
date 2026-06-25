# RULER-Lite Byte Tasks

Status: implemented as the next controlled long-context stress test after the
single byte-needle task.

## Why This Exists

The single needle task proves that the router can find one distant span. That is
not enough. RULER-style tasks add pressure in three ways:

```text
many possible needles
chains of lookup
answers assembled from multiple distant regions
```

This keeps the benchmark controlled while moving closer to long-context
retrieval behavior that matters in larger LLMs.

## Implemented Tasks

### Multi-Needle

CLI:

```text
--task ruler-multineedle
```

The context contains many key-value records:

```text
KEY ABCD VALUE 7XQ9
KEY W2LM VALUE KJ4T
...
QUESTION: value for KEY ABCD?
ANSWER:
```

The model must pick the requested key and emit its value.

### Multi-Hop

CLI:

```text
--task ruler-multihop
```

The context contains link records and value records:

```text
LINK A123 -> B777
VALUE B777 = Z9QK
...
QUESTION: follow KEY A123.
ANSWER:
```

The model must use the first lookup to find the second lookup key, then retrieve
the final value.

### Aggregation

CLI:

```text
--task ruler-aggregation
```

The context contains several ordered parts for one group plus distractors:

```text
GROUP A123 PART 00 = QWER
GROUP Z999 PART 00 = KJHG
GROUP A123 PART 01 = TYUI
...
QUESTION: concatenate GROUP A123 in part order.
ANSWER:
```

The model must retrieve multiple distant records and concatenate their values in
part order.

## Shared Controls

```text
--ruler-num-needles         total records/chains to place in the context
--ruler-key-length          byte length of generated keys
--ruler-aggregation-count   number of target parts for aggregation
--needle-code-length        byte length of each value code
--needle-min-gap            minimum byte gap from source value to answer
```

The answer bytes use the same restricted alphabet as the original needle task,
so `answer_random_chance` remains `1 / 36` per answer byte.

## Current MemoryCache Recipe

Use the MemoryCache runner:

```bash
python scripts/run_memory_cache_ruler.py \
  --task ruler-multineedle \
  --seq-len 4096 \
  --needle-min-gap 3072
```

The dataset is implemented in `src/uabla/ruler_lite.py` so MemoryCache can use
the same controlled long-context tasks without mutating the byte-LM training
path.

## Recommended Evaluation Order

Run in this order:

```text
1. ruler-multineedle at 4096 bytes, 8 needles
2. ruler-multineedle at 4096 bytes, 16 needles
3. ruler-multihop at 4096 bytes, 8 chains
4. ruler-aggregation at 4096 bytes, 8 records, 3 target parts
5. repeat the survivors at 8192 bytes
```

Track:

```text
source_hit_rate
source_set_hit_rate for aggregation
mean_active_exact_tokens
active_token_compression_ratio
tokens_per_second
```
