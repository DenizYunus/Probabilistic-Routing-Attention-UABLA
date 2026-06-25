# UABLA Memory Cache Alternative

Status: experimental inference-memory track inside the main UABLA repo.

## Goal

Reduce active KV memory without turning old context into permanent forgetting.

The key distinction:

```text
evicted from GPU exact KV != forgotten
```

Old context can remain available as summaries and raw token spans, then exact
leaves can be revived only when the router asks for them.

## Current Prototype

Implemented in `src/uabla/memory_cache.py`:

```text
RevivalMemoryCache
MultiSlotLeafCompressor
AverageLeafCompressor
HybridLeafCompressor
MemoryCacheRouteProbe
UsefulnessLedger
scheduled_recent_window
source-hit metrics
top-k survival loss
```

The simulator accepts:

```text
hidden_states: [batch, seq_len, hidden]
query_states:  [batch, queries, hidden]
```

and returns:

```text
candidate_indices
candidate_mask
revived_leaf_indices
leaf_scores
leaf_summaries
active_exact_tokens_per_query
```

## Memory Shape

```text
Recent exact cache
  last N tokens, always exact

Old summary cache
  older tokens split into leaves
  each leaf compressed into multiple learned slots

Revived exact leaves
  exact old token spans selected by summary routing

Usefulness ledger
  tracks which leaves get revived over time
```

## Why Multi-Slot Summaries

One average can blur important details. Multi-slot summaries let a leaf store
several views of the same chunk:

```text
slot 1: one salient pattern
slot 2: another salient pattern
slot 3: exact-ish symbol/date/number evidence
slot 4: residual semantics
```

The code does not hard-code those meanings. It gives the compressor multiple
learned slots and lets training shape them.

## Multi-Slot V2

The first tight-budget sweep showed a useful surprise:

```text
average summaries were stronger than naive multi-slot summaries
```

That does not mean multi-slot is a bad idea. It likely means the first scoring
rule was too brittle. With `max` scoring, only the best slot in a leaf receives
strong routing pressure, so the other slots can collapse or become unused.

The v2 experiment keeps the same memory shape but adds two controls:

```text
slot_score_reduction = max | logsumexp | mean
summary_diversity_weight
```

The recommended v2 setting is:

```text
slot_score_reduction = logsumexp
slot_score_temperature = 0.7
summary_diversity_weight = 0.01
```

`logsumexp` is a smooth version of max. It still rewards the leaf when any slot
matches the query, but it gives gradients to more than one slot. The diversity
loss lightly discourages all summary slots in a leaf from becoming the same
vector.

## Hybrid Summary V3

The next result was also useful:

```text
logsumexp rescued multi-slot, but average summaries still won the main sweep
```

So v3 keeps the same active-token budget but changes the summary composition:

```text
slot 0: deterministic average anchor
slot 1..N: learned salient/residual slots
```

This keeps the stable signal that made the average baseline strong, while still
giving the router extra learned slots for details that an average can blur. Use:

```bash
python scripts/run_memory_cache_ruler.py \
  --compressor-type hybrid \
  --summary-slots 4 \
  --slot-score-reduction logsumexp \
  --slot-score-temperature 0.7 \
  --summary-diversity-weight 0.01
```

## Dual Router V1

The hybrid sweep showed that learned slots still did not beat the average
summary. That pushed the design in a different direction:

```text
average/neural address + symbolic byte-sketch address + uncertainty-gated top-k
```

The symbolic path builds a tiny hashed n-gram sketch for each old leaf and for
the query window. It is not meant to replace neural routing. It gives the cache
a cheap exact-ish signal for keys, names, numbers, codes, and repeated byte
patterns that dense vectors can blur.

The dynamic top-k path keeps a minimum number of revived leaves, then opens
extra leaves only when their scores are within a margin of the best leaf:

```text
sharp route       -> revive min_k leaves
uncertain route   -> revive more leaves up to top_k
```

The locked v1.2 quality setting is:

```bash
python scripts/run_memory_cache_ruler.py \
  --compressor-type average \
  --summary-slots 1 \
  --revived-top-k 4 \
  --symbolic-weight 8.0 \
  --symbolic-hash-bins 2048 \
  --symbolic-ngrams 2,3,4
```

The locked v1.2 efficient setting is:

```bash
python scripts/run_memory_cache_ruler.py \
  --compressor-type average \
  --summary-slots 1 \
  --revived-top-k 3 \
  --symbolic-weight 8.0 \
  --symbolic-hash-bins 2048 \
  --symbolic-ngrams 2,3,4 \
  --symbolic-ngram-weights 0.25,1.0,0.5
```

On RULER-lite byte multi-needle at 4096 bytes, the quality setting improved
source-hit recall from 57.12% to 71.38% with the same 320 active tokens per
query. The efficient setting reached 67.63% with 304 active tokens. See:

```text
docs/MEMORY_CACHE_DUAL_ROUTER_V1_LOCKED.md
```

Dynamic top-k remains experimental. Fixed top-k profiles are the locked v1.2
path.

## Multi-Ngram Symbolic Routing

The locked v1.2 profile uses several symbolic sketches:

```text
symbolic_ngrams = 2,3,4
```

Equal weights are locked for the quality profile:

```bash
python scripts/run_memory_cache_ruler.py \
  --symbolic-ngrams 2,3,4
```

Weighted sketches are locked for the efficient top-3 profile:

```bash
python scripts/run_memory_cache_ruler.py \
  --symbolic-ngrams 2,3,4 \
  --symbolic-ngram-weights 0.25,1.0,0.5
```

The combined symbolic score is a weighted average of each n-gram sketch score.
This keeps the locked single-ngram behavior available while allowing a broader
index:

```text
2-grams: boundary-tolerant fuzzy matching
3-grams: current locked exact-ish signal
4-grams: stricter code/key matching
```

Diagnostics showed the symbolic path is a reranking bias, not a standalone
router: symbolic top-k hit stayed near 9-11%, but combined rank improved
enough to lift source-hit recall.

## Answer-Set Supervision

Aggregation tasks need several source records to survive. Per-token source
supervision can under-train that behavior because every answer byte points at
one source byte. The runner now supports an opt-in aggregation target mode:

```bash
python scripts/run_memory_cache_ruler.py \
  --task ruler-aggregation \
  --source-target-mode answer-set
```

In answer-set mode, every answer query targets the full set of answer source
leaves. The logged `source_hit_rate` becomes source-leaf recall over that set.
This is experimental and should be compared against the locked V1.2 profiles.

Answer-set mode also supports an explicit coverage objective:

```bash
python scripts/run_memory_cache_ruler.py \
  --task ruler-aggregation \
  --source-target-mode answer-set \
  --source-coverage-weight 0.2 \
  --source-coverage-temperature 1.0
```

The original multi-positive set loss can be satisfied by one strong positive
leaf. Coverage loss uses a uniform distribution over all source leaves, so it
penalizes routes that collapse onto only one part of an aggregation answer.

Dual-target mode keeps the strong V1.2 per-token objective while adding
answer-set coverage as auxiliary pressure:

```bash
python scripts/run_memory_cache_ruler.py \
  --task ruler-aggregation \
  --source-target-mode dual \
  --source-coverage-weight 0.05
```

Dual logs use:

```text
source_hit_rate       per-token exact-source survival
source_set_hit_rate   answer-set source-leaf recall
```

Dual `0.05` is locked as the aggregation-friendly profile:

```text
per-token anchor: source_hit_rate 57.95%
dual cov 0.05:   source_hit_rate 57.82%, source_set_hit_rate 41.49%
```

Higher coverage weights slightly improved set-hit but damaged per-token
retrieval, so they remain task-specific experiments rather than the default.

## Diverse Revival

The next experimental selector is MMR-style diverse revival:

```bash
python scripts/run_memory_cache_ruler.py \
  --revival-selection diverse \
  --diversity-weight 0.2 \
  --diversity-candidate-multiplier 4
```

Plain top-k can spend several revival slots on near-duplicate leaf summaries.
Diverse revival first builds a high-score candidate pool, selects the best leaf,
then subtracts a similarity penalty from later candidates:

```text
adjusted_score = route_score - diversity_weight * max_similarity_to_selected
```

Similarity is cosine similarity between average leaf-summary vectors and is
clamped at zero, so dissimilar leaves are not artificially boosted. The goal is
to improve aggregation and multi-hop coverage without increasing active exact
tokens.

## Three Experimental Revival Ideas

These are not locked yet. They are intended to be tested beside the current
dual-router aggregation profile.

### Entropy-Adaptive Budget

Fixed `revived_top_k` spends the same exact-token budget on every query. Some
queries have a sharp route distribution and probably need fewer old leaves; some
queries are uncertain and should be allowed to open more.

```bash
python scripts/run_memory_cache_ruler.py \
  --revived-top-k 6 \
  --adaptive-budget-mode entropy \
  --adaptive-min-k 4 \
  --adaptive-entropy-threshold 0.75
```

The router computes normalized entropy over leaf scores. Low-entropy queries
keep only `adaptive_min_k`; high-entropy queries keep the full top-k. This makes
the exact-token budget input-dependent.

### Channel-Mix Revival

The combined score can hide disagreement between channels. If the neural route
and symbolic route each see different useful evidence, pure top-k may let one
channel dominate all slots.

```bash
python scripts/run_memory_cache_ruler.py \
  --revival-selection channel-mix \
  --channel-mix-neural-k 1 \
  --channel-mix-symbolic-k 1
```

This reserves a small number of candidate slots for neural-only and
symbolic-only top leaves, then fills the rest from combined-score order. It is a
cheap hedge against channel collapse.

### Revived-Leaf Diversity Loss

Summary-slot diversity works inside a leaf. This loss works across the leaves
selected for a query:

```bash
python scripts/run_memory_cache_ruler.py \
  --revived-diversity-weight 0.02
```

It penalizes positive cosine similarity between revived leaf summaries. The
intent is not to force random diversity; it only lightly discourages spending
several exact revival slots on nearly identical old chunks.

## Selector-Aware Training Experiments

The first adaptive-budget attempt pruned after training. That was too harsh:
the scorer had been trained to keep the source leaf inside top-4, then inference
sometimes asked it to survive inside top-2. The next experiment trains that
pressure directly:

```bash
python scripts/run_memory_cache_ruler.py \
  --budget-ladder-weight 0.5 \
  --budget-ladder-top-ks 2,3,4 \
  --adaptive-budget-mode entropy \
  --adaptive-min-k 2
```

`budget_ladder_loss` averages survival losses across the listed budgets. The
goal is a scorer whose top leaves are more stable under cache pressure, so
adaptive pruning becomes a trained behavior instead of an inference trick.

## Neural-Gated Symbolic Scoring

The symbolic route is useful but noisy. Reserving symbolic slots hurt because it
could displace stronger neural/combined candidates. A softer version keeps the
symbolic rerank signal but gates it through neural plausibility:

```bash
python scripts/run_memory_cache_ruler.py \
  --symbolic-gate-top-k 16
```

Only leaves inside the neural top-k receive symbolic score. This tests whether
symbolic sketches are best used as a local reranker rather than a global rescue
channel.

## Rank-Fusion Revival

Raw neural and symbolic scores live on different scales. Rank fusion ignores raw
magnitude and combines reciprocal ranks:

```bash
python scripts/run_memory_cache_ruler.py \
  --revival-selection rank-fusion \
  --rank-fusion-neural-weight 1.0 \
  --rank-fusion-symbolic-weight 1.0 \
  --rank-fusion-combined-weight 1.0 \
  --rank-fusion-offset 8.0
```

This is an inference selector: losses still train the combined leaf scores, but
revival can prefer leaves that multiple channels rank well even when their raw
scores are not directly comparable.

## Gate-Top8 Follow-Up Ideas

The gate sweep made the current best principle clearer:

```text
use symbolic routing as a local reranker inside neural-plausible leaves
```

The top8 gate improved source-hit without increasing active exact tokens. The
next ideas test whether that principle can be sharpened.

### Neural-Gate Auxiliary Loss

If the symbolic scorer only sees neural top8 leaves, then the neural scorer must
learn to put source leaves inside that pool. This auxiliary loss trains that
directly:

```bash
python scripts/run_memory_cache_ruler.py \
  --symbolic-gate-top-k 8 \
  --neural-gate-loss-weight 0.25 \
  --neural-gate-top-k 8
```

The auxiliary loss is a top-k survival loss on `neural_leaf_scores`, while the
main loss still trains the combined scorer.

### Explicit Gated Rerank

Plain symbolic gating zeros symbolic score outside the neural pool, but final
selection can still choose high-neural leaves outside that pool. Explicit
gated-rerank makes the final selection two-stage:

```text
1. take neural top8 leaves
2. select final top4 by combined neural + symbolic score inside that pool
```

Run it with:

```bash
python scripts/run_memory_cache_ruler.py \
  --symbolic-gate-top-k 8 \
  --revival-selection gated-rerank \
  --gated-rerank-top-k 8
```

### Symbolic Score Normalization

Neural scores and symbolic sketch scores do not naturally live on the same
scale. Score normalization tests whether symbolic routing should contribute as a
calibrated rerank signal rather than a large raw additive bonus:

```bash
python scripts/run_memory_cache_ruler.py \
  --symbolic-gate-top-k 8 \
  --symbolic-score-norm zscore \
  --symbolic-weight 2.0
```

Supported symbolic normalization modes are `none`, `zscore`, and `minmax`.

## V1.4 Address-Context Breakthrough

The first profile family to clear the 80% KPI adds address context around each
exact payload leaf:

```bash
python scripts/run_memory_cache_ruler.py \
  --address-context-left 32 \
  --address-context-right 4 \
  --symbolic-context-left 32 \
  --symbolic-context-right 4 \
  --neural-gate-loss-weight 0.25
```

This lets summaries and symbolic sketches see nearby record metadata while the
active exact cache still revives only the compact payload leaf.

Locked profiles:

```text
efficient:
  leaf_size = 8
  revived_top_k = 8
  active exact tokens = 320
  source_hit_rate = 89.48%
  source_set_hit_rate = 60.17%

quality:
  leaf_size = 16
  revived_top_k = 8
  active exact tokens = 384
  source_hit_rate = 94.99%
  source_set_hit_rate = 74.72%
```

See `docs/MEMORY_CACHE_V1_4_LOCKED.md`.

## Dual-Score Diagnostics

Eval and train logs now include score-channel diagnostics when available:

```text
combined_source_rank
neural_source_rank
symbolic_source_rank
neural_source_hit_rate
symbolic_source_hit_rate
```

These tell us whether a run improves because the symbolic route finds leaves
the neural route missed, or because it only boosts leaves that were already
easy. This is the main tool for deciding whether new symbolic tricks are real.

## Cache Diet Curriculum

The cache should be trained under pressure:

```text
start: large exact recent window
end: smaller exact recent window
old context must survive through summaries + revival
```

The first helper is:

```text
scheduled_recent_window(start_window, end_window, step, curriculum_steps)
```

## First Success Criteria

Before integrating with full autoregressive KV cache:

```text
1. candidate set includes recent tokens exactly
2. summary routing revives the correct old leaf in toy tests
3. active exact tokens are far fewer than full sequence length
4. source-hit survival can be measured on RULER-lite
5. multi-slot summaries beat average summaries in source-hit survival
```

Items 1-4 now have code/tests. Item 5 is tested with three comparisons:

```bash
python scripts/run_memory_cache_ruler.py --compressor-type average ...
python scripts/run_memory_cache_ruler.py --compressor-type multislot --slot-score-reduction max ...
python scripts/run_memory_cache_ruler.py --compressor-type multislot --slot-score-reduction logsumexp --summary-diversity-weight 0.01 ...
python scripts/run_memory_cache_ruler.py --compressor-type hybrid --slot-score-reduction logsumexp --summary-diversity-weight 0.01 ...
python scripts/run_memory_cache_ruler.py --compressor-type average --symbolic-weight 8.0 --symbolic-hash-bins 2048 ...
```

The main metric is:

```text
source_hit_rate
```

It measures whether the exact source byte is present in the active candidate
set after keeping recent tokens and reviving selected old leaves.

The default training objective now combines:

```text
0.2 * source_leaf_cross_entropy
+ 1.0 * top_k_survival_loss
+ optional summary_diversity_weight * summary_diversity_loss
```

The survival loss directly optimizes the actual cache behavior: the source leaf
only needs to outrank the kth strongest non-source leaf, because revival keeps a
top-k set rather than only the top-1 leaf.

## Next Implementation Steps

1. Validate locked profiles on aggregation and multi-hop RULER-lite.
2. Increase task hardness: more needles, longer key/value codes, larger gaps.
3. Test longer contexts: 8192 bytes, then 16384 bytes.
4. Add cache-diet training: progressively reduce recent window.
5. Add a trained source-hit checkpoint as a warm start for UABLA fusion.
6. Later, implement true streaming KV cache with CPU-side raw-token revival.

## Non-Goal For Now

This branch does not yet implement production autoregressive serving. It is a
controlled simulator for proving the memory policy before the cache engine gets
expensive and stateful.
