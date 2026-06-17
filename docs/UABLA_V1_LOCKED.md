# UABLA-448 V1 Locked Architecture

Status: locked for V1 implementation.

## One-Line Definition

UABLA stores each previous token as a compact probabilistic memory, routes through four-centroid superblock and block summaries, opens dense parent blocks, and uses query uncertainty to choose how much context to retrieve.

## Core Claim

Replace full key/value cache with:

```text
cache_t = [mu_store, log_sigma_store, compressed_value, position]
```

Then use:

```text
local window
+ four-centroid superblock routing
+ four-centroid block routing
+ half-stride shifted block routing summaries
+ adaptive bucketed token selection
```

to reduce cache memory and long-context attention compute.

## V1 Non-Negotiables

1. Use compressed probabilistic store memories, not full cached K/V.
2. Use four centroids per block.
3. Use multiscale four-centroid superblock routing before block routing.
4. Route against block centroids after superblock narrowing.
5. Route over both regular and half-stride shifted block summaries.
6. Open deduplicated dense parent blocks after centroid routing.
7. Expand selected routed tokens with a causal successor-biased byte span.
8. Always include a local causal window.
9. Use bucketed top-k, not true top-p, in V1.
10. Use a cheap diagonal Gaussian distance plus optional low-rank hybrid scoring.
11. Use straight-through differentiable sparse routing.
12. Support training-only dense-attention teacher distillation.
13. Keep memory importance within the existing 448-dim cache budget.
14. Keep the implementation causal.
15. Never materialize a full `T x T` distance matrix in serious paths.
16. Treat this as an attention/cache research prototype, not a DeepSeek-scale training recipe.

## Validated Byte-Needle V1 Recipe

Status: locked as the current validated training recipe for byte-level
needle-in-haystack experiments.

Use the named CLI recipe:

```bash
python scripts/run_byte_lm.py --recipe uabla-v1-byte-needle ...
```

The recipe locks the training-only routing scaffold to:

```text
direct_route_weight = 0.0
token_contrast_weight = 0.0005
token_contrast_warmup_steps = 1000
token_contrast_decay_steps = 3500
token_contrast_temperature = 0.7
route_entropy_weight = 0.02
route_entropy_min = 0.35
route_entropy_decay_steps = 6500
route_budget_curriculum = explore
route_budget_curriculum_boundaries = 2500,6500
stage_shifted_when_answer_accuracy = 0.5
stage_shifted_min_step = 6501
local_window = 128
byte_route_patch_size = 16
routed_span_left = 2
routed_span_right = 8
```

The budget curriculum is:

```text
step 1:    superblock [4, 8], centroid [16, 32], token [32, 64]
step 2501: superblock [2, 4], centroid [8, 16],  token [16, 32]
step 6501: superblock [2, 4], centroid [4, 8],  token [8, 16]
```

The validated floor on TinyStories byte needle recall at sequence length 1024:

| Token contrast weight | Final answer accuracy | Result |
| ---: | ---: | --- |
| 0.0005 | 99.9674% | locked V1 setting |
| 0.0002 | 99.7721% | works, but near the cliff |
| 0.0 | 9.1309% | fails |

Interpretation:

```text
direct route supervision is not part of the locked V1 recipe
token contrast is a tiny training-only retrieval hint
inference uses no auxiliary loss
cache stays at the configured UABLA cache dimension
```

## Locked Dimensions

UABLA-448 cache dimensions per token per layer:

| Component | Dimension |
| --- | ---: |
| Store mean `mu_store` | 64 |
| Store log sigma `log_sigma_store` | 64 |
| Compressed value `compressed_value` | 256 |
| Position component `position` | 64 |
| Total cached dim | 448 |

The current query-side seek distribution is not cached:

| Component | Dimension |
| --- | ---: |
| Seek mean `mu_seek` | 64 |
| Seek log sigma `log_sigma_seek` | 64 |

## Block Layout

V1 block settings:

```text
block_size = 128
centroids_per_block = 4
tokens_per_centroid_average = 32
superblock_size_blocks = 8
use_shifted_blocks = true
shifted_block_offset = block_size / 2
```

Each block owns four centroid distributions:

```text
S[b, c] = Normal(mu_block[b, c], diag(sigma_block[b, c]^2))
```

where:

```text
b = block index
c = centroid index in {0, 1, 2, 3}
```

V1 also builds a second shifted routing grid:

```text
regular blocks:
[0..127]     [128..255]     [256..383] ...

shifted blocks:
      [64..191]      [192..319]      [320..447] ...
```

The shifted grid exists only as routing summaries. Token cache entries are not
duplicated, so shifted routing does not change the per-token cache dimension.
Regular and shifted block centroids compete in one global centroid top-k, so
the final routed candidate width stays fixed instead of doubling.

## Centroid Construction

V1 uses learned soft assignment from store means:

```text
assign_logits = W_assign(mu_store)
assign_weights = softmax(assign_logits over 4 centroids)
```

For each block and centroid:

```text
w_jc = assignment weight of token j to centroid c
```

Centroid mean:

```text
mu_block[c] = sum_j w_jc * mu_store[j] / (sum_j w_jc + eps)
```

Centroid variance:

```text
var_block[c] =
  sum_j w_jc * (sigma_store[j]^2 + (mu_store[j] - mu_block[c])^2)
  / (sum_j w_jc + eps)
```

Store the centroid uncertainty as:

```text
log_sigma_block[c] = 0.5 * log(var_block[c] + eps)
```

Rationale:

```text
K = 1 hides mixed-topic blocks.
K = 4 gives multi-topic summaries without collapsing back into token search.
K = 128 is token-level routing again and is out of scope.
```

## Superblock Routing

V1 adds one hierarchy level above blocks:

```text
8 blocks -> 1 superblock
each superblock -> 4 centroid distributions
```

Superblock centroids are built by weighted aggregation of their child block centroids, preserving:

```text
mean of child centroid memories
+ uncertainty/spread across child blocks
```

Routing order:

```text
1. score routeable regular and shifted superblock centroids
2. select adaptive superblock hits per grid
3. restrict block-centroid routing to opened superblocks plus recent tail blocks
4. select a global top-k across regular and shifted block centroids
5. open deduplicated parent token ranges
6. select tokens inside gathered candidates
```

Default superblock-hit buckets:

```text
superblock_hit_buckets = [2, 4, 8, 16]
```

This is the first scaling step toward:

```text
region -> block -> token
```

without changing token cache size.

## Routing Distance

V1 uses the cheap diagonal Gaussian distance:

```text
D(q, s) =
  ||(mu_q - mu_s) / (sigma_q + sigma_s + eps)||_2^2
  + alpha * ||log_sigma_q - log_sigma_s||_1
```

Score:

```text
score = -D(q, s) / tau + position_bias
```

Default constants:

```text
alpha = 0.1
tau = 1.0
min_log_sigma = -5.0
max_log_sigma = 2.0
eps = 1e-6
```

V1 may also add a low-rank learned compatibility term:

```text
score += hybrid_scale * dot(W_hq mu_seek, W_hs mu_store) / sqrt(rank)
```

Default:

```text
hybrid_score_rank = 16
hybrid_score_scale = 0.1
```

This adds small parameters and candidate-time compute only. It does not add cache memory.

## Uncertainty Signal

Query uncertainty:

```text
u_t = mean(log_sigma_seek[t])
```

High uncertainty opens more context. Low uncertainty opens less context.

## Routing Budgets

V1 uses bucketed top-k budgets.

Centroid-hit budget:

```text
centroid_hit_buckets = [4, 8, 16, 32]
```

Token-selection budget:

```text
token_k_buckets = [16, 32, 64, 128]
```

Routed token span expansion:

```text
routed_span_left = 2
routed_span_right = 8
```

After hard token selection, V1 opens a small causal span around selected routed
positions before the final attention softmax. The successor bias helps
byte-level copy tasks when the router lands inside a word or code rather than
on the exact first byte. This span does not add cache memory.

The bucket index is chosen from query uncertainty:

```text
bucket = sigmoid((u_t - beta) / gamma)
```

V1 defaults:

```text
beta = -1.5
gamma = 0.5
```

## Differentiable Sparse Routing

V1 does not use plain hard top-k as a dead-end router. It uses straight-through sparse routing:

```text
forward pass:
  hard bucketed top-k centroid hits
  hard opened parent blocks
  hard bucketed top-k token attention

backward pass:
  softmax gradients over routeable centroids
  softmax gradients over candidate tokens
```

Centroid routing computes:

```text
soft_centroid = softmax(route_scores over routeable centroids)
hard_centroid = topk_mask(route_scores, centroid_budget)
```

Then uses the straight-through estimator:

```text
st_centroid = hard_centroid + soft_centroid - stop_gradient(soft_centroid)
```

The four centroid probabilities are summed into a parent block prior:

```text
soft_block = sum_centroids(soft_centroid)
hard_block = opened_parent_block_mask
st_block = hard_block + soft_block - stop_gradient(soft_block)
```

Opened candidate tokens receive a differentiable block prior:

```text
token_score += log(st_block[parent_block] + eps)
```

Since `hard_block = 1` for opened blocks in the forward pass, this prior does not distort the sparse inference path. In the backward pass, it gives route scores, centroid summaries, and assignment weights a learning signal.

Token selection uses the same principle:

```text
attn_hard = softmax(scores masked to bucketed top-k)
attn_soft = softmax(scores over all candidates)
attn = attn_hard + attn_soft - stop_gradient(attn_soft)
```

So the forward pass is sparse, while the backward pass trains token scoring over all gathered candidates.

## Teacher-Distilled Routing

Dense teacher attention is useful for training the router, but it is not part of inference.

Training-only distillation:

```text
teacher attention mass over tokens
-> sum into block or superblock regions
-> cross-entropy against UABLA route probabilities
```

This teaches the router where dense attention would have spent probability mass.

Important product constraint:

```text
teacher distillation adds zero inference cache memory
teacher distillation can be disabled at inference
teacher distillation can be run on short windows, sampled layers, or offline batches
```

The implementation exposes route scores only when `return_routing=True`.

## Memory Importance

V1 includes an optional learned memory-importance gate:

```text
memory_importance_t = sigmoid(W_importance x_t)
```

It affects:

```text
centroid construction weights
candidate token score bias
```

To keep the cache target intact, the raw importance logit is stored in the existing 64-dim position payload:

```text
position[0] = memory_importance_logit
```

So the cache remains:

```text
64 + 64 + 256 + 64 = 448 dims/token/layer
```

## Candidate Set

For each token, candidates are:

```text
local causal window
+ dense parent blocks opened by selected centroid hits
+ optional global memory tokens
```

Locked V1 defaults:

```text
local_window = 256
selected_centroid_hits = adaptive [4, 8, 16, 32]
opened_blocks = deduplicated parent blocks
token_selection = adaptive [16, 32, 64, 128]
global_memory_tokens = 0 in the first code prototype
```

Global memory tokens are reserved for V1.1, not the first implementation.

## Attention Output

Inside the final selected candidate tokens:

```text
scores[t, j] = -D(seek[t], store[j]) / tau + position_bias(t, j)
attn[t, j] = softmax(scores[t, selected_j])
value[j] = W_value_up(compressed_value[j])
out[t] = sum_j attn[t, j] * value[j]
y[t] = W_out(out[t])
```

## Position Handling

V1 includes a small cached position component for parity with the 448-dim cache target, but the first code path may use a simple learned relative-position bias for scoring.

Default position bias:

```text
b_pos(t, j) = learned_bucket(t - j)
```

Local causal order is protected by always including the local window.

## Training Schedule

Phase 1:

```text
fixed budgets
straight-through centroid routing enabled
sigma predicted but does not control budgets
```

Phase 2:

```text
fixed centroid routing
adaptive token budget enabled
```

Phase 3:

```text
adaptive centroid-hit budget enabled
```

Phase 4:

```text
budget regularization added slowly
```

## Losses

Base loss:

```text
language_model_loss
```

Optional V1 regularizers:

```text
sigma range penalty
teacher routing distillation
guarded budget penalty
router entropy monitor
```

Guarded budget regularization penalizes large budgets mostly when the model is already prediction-confident:

```text
budget_loss = confidence_guard * budget_fraction
```

This avoids teaching the model to be cheap when it still needs context.

## Deferred From First Implementation

These are intentionally not part of the first build:

1. True top-p routing.
2. ANN retrieval.
3. External memory.
4. Custom block-sparse CUDA/Triton kernels.
5. Global memory tokens.
6. Multi-layer cache manager optimized for generation.
7. MLA baseline reproduction.
8. DeepSeek-scale dimensions beyond config support.

## First Implementation Target

Build a PyTorch module that supports:

```text
batch_first input: [batch, seq, hidden]
causal self-attention
UABLA-448 defaults
configurable small dimensions for tests
four-centroid block summaries
local window
centroid routing
parent-block opening
adaptive bucketed token selection
```

The first test suite should verify:

```text
output shape
causal masking
centroid tensor shapes
cache dim equals configured total
budget buckets increase with uncertainty
no full T x T matrix is created by the main candidate builder
```

## Locked V1 Name

```text
UABLA-448
Uncertainty-Adaptive Block Latent Attention with four-centroid block routing
```
