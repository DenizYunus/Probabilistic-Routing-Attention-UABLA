# Probabilistic Routing Attention

This repository contains a clean V1 prototype of **UABLA-448** and a sibling
MemoryCache track:

```text
Uncertainty-Adaptive Block Latent Attention
```

The architecture is locked in:

[docs/UABLA_V1_LOCKED.md](docs/UABLA_V1_LOCKED.md)

Implementation work should follow that document as the source of truth.

Current validated byte-needle V1 recipe:

```bash
PYTHONPATH=src python scripts/run_byte_lm.py \
  --recipe uabla-v1-byte-needle \
  --text-file data/tinystories_combo.txt \
  --seq-len 1024 \
  --needle-min-gap 768 \
  --steps 10000 \
  --batch-size 4 \
  --lm-loss-weight 0.2 \
  --answer-loss-weight 5.0 \
  --amp
```

This locks contrast-only routing help at `token_contrast_weight=0.0005`,
with no direct route supervision. The latest TinyStories byte-needle floor was:
`0.0005` works, `0.0002` is close to the cliff, and `0.0` fails.

The first test plan is here:

[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)

Current locked byte-LM RULER rerank result:

[docs/RULER_RERANK_V1_LOCKED.md](docs/RULER_RERANK_V1_LOCKED.md)

The calibrated 4096-token tokenized RULER multi-needle rerank checkpoint reached
`92.76%` answer accuracy and `58.79%` exact match with about `10.17GB` peak
training memory. This locks pure candidate reranking as the current best
byte-LM answer-decoding path.

## MemoryCache Track

MemoryCache is the separate inference-memory/cache-revival line inside this
repo. It asks a different question from UABLA:

```text
UABLA:       where should this query look?
MemoryCache: what old context stays exact, compressed, revivable, or evicted?
```

Current locked MemoryCache profile:

[docs/MEMORY_CACHE_V1_4_LOCKED.md](docs/MEMORY_CACHE_V1_4_LOCKED.md)

The current V1.4 result on RULER-lite byte aggregation at 4096 bytes:

```text
efficient profile: source_hit_rate 89.48%, active exact 320, compression 12.8x
quality profile:   source_hit_rate 94.99%, active exact 384, compression 10.67x
```

MemoryCache source code and runner:

```text
src/uabla/memory_cache.py
src/uabla/ruler_lite.py
scripts/run_memory_cache_ruler.py
```
