# UABLA Experiment Plan

This plan starts with synthetic long-memory tasks before moving to real text.

## Why Not Wikipedia First

Wikipedia is useful later, but it hides routing failures under real-language noise. First we need crisp tasks where the correct memory access is obvious.

## First Task: Key-Value Recall

Sequence format:

```text
key value sep key value sep ... filler ... QUERY key value
```

The model receives everything except the final value and must predict that value from the earlier matching key-value pair.

Multi-query variant:

```text
key value sep ... filler ... QUERY key value QUERY key value ...
```

Use `--num-queries N` to add several supervised retrieval targets per sequence. This is the preferred debugging task after the first 512-token runs, because the single-query setup gives only one answer gradient per long sequence.

Primary metric:

```text
answer_accuracy
```

Secondary metrics:

```text
answer_loss
tokens_per_second
cache_dim_per_token_per_layer
route_entropy
avg_open_blocks
avg_token_budget
attention_entropy
```

## Baseline Runs

Dense:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention dense \
  --seq-len 128 \
  --num-pairs 8 \
  --num-queries 4 \
  --steps 500 \
  --batch-size 16
```

Local:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention local \
  --seq-len 128 \
  --num-pairs 8 \
  --num-queries 4 \
  --local-window 32 \
  --steps 500 \
  --batch-size 16
```

UABLA:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention uabla \
  --seq-len 128 \
  --num-pairs 8 \
  --num-queries 4 \
  --steps 500 \
  --batch-size 16
```

## Teacher-Distilled Routing

Train and save a dense teacher:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention dense \
  --seq-len 128 \
  --num-pairs 8 \
  --steps 1000 \
  --batch-size 16 \
  --save-checkpoint runs/dense_teacher.pt
```

Train UABLA with training-only teacher routing distillation:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention uabla \
  --seq-len 128 \
  --num-pairs 8 \
  --steps 1000 \
  --batch-size 16 \
  --teacher-checkpoint runs/dense_teacher.pt \
  --distill-weight 0.1
```

Budget pressure experiment:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention uabla \
  --seq-len 128 \
  --num-pairs 8 \
  --steps 1000 \
  --batch-size 16 \
  --teacher-checkpoint runs/dense_teacher.pt \
  --distill-weight 0.1 \
  --budget-weight 0.01
```

Direct routing supervision:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention uabla \
  --seq-len 256 \
  --num-pairs 16 \
  --num-queries 8 \
  --steps 10000 \
  --batch-size 8 \
  --teacher-checkpoint runs/dense_teacher_256_mq.pt \
  --distill-weight 0.3 \
  --direct-route-weight 0.5 \
  --budget-weight 0.0
```

This synthetic-only loss uses the known source value position for each query
and trains the router to place probability mass on the matching block and
superblock. It does not change inference.

Token contrastive retrieval curriculum:

```bash
PYTHONPATH=src python scripts/run_synthetic.py \
  --attention uabla \
  --seq-len 256 \
  --num-pairs 16 \
  --num-queries 8 \
  --steps 10000 \
  --batch-size 8 \
  --teacher-checkpoint runs/dense_teacher_256_mq.pt \
  --distill-weight 0.3 \
  --direct-route-weight 0.05 \
  --token-contrast-weight 0.2 \
  --token-contrast-warmup-steps 500 \
  --token-contrast-decay-steps 4000 \
  --token-contrast-temperature 0.7 \
  --budget-weight 0.0
```

This loss ranks the true source token above other retrieved candidates in the
final UABLA layer. It only applies when the source token is already in the
candidate set, so it pairs naturally with a small direct routing weight. The
warmup/decay schedule treats token labels as a temporary curriculum rather than
a permanent oracle.

## Success Criteria For This Stage

UABLA should:

```text
beat local attention on answer_accuracy
approach dense answer_accuracy
use lower cache_dim_per_token_per_layer than dense
show non-collapsed route_entropy
show adaptive routing diagnostics once adaptive schedules are enabled
```

If UABLA fails key-value recall, do not move to Wikipedia yet.

## Next Tasks

Add synthetic tasks in this order:

```text
needle-in-haystack
two-hop key-value lookup
topic-switch retrieval
copy from fixed far offset
code dependency recall
```

Then move to real text:

```text
TinyStories
WikiText
PG-19/books
small code corpus
Wikipedia dumps or Hugging Face Wikipedia datasets
```

## Tokenizer-Free Byte LM Track

Byte-level language modeling is the first real-text bridge. It uses raw UTF-8
byte IDs `0..255` plus reserved special IDs, so there is no BPE tokenizer. A
causal local convolutional mixer can be enabled before attention with
`--byte-mixer-kernel`. UABLA keeps byte-level next-token prediction but routes
over latent byte chunks controlled by `--byte-route-patch-size`.

Dense byte LM:

```bash
PYTHONPATH=src python scripts/run_byte_lm.py \
  --attention dense \
  --text-file data/wiki_sample.txt \
  --seq-len 512 \
  --steps 5000 \
  --batch-size 8 \
  --hidden-size 128 \
  --layers 4 \
  --byte-mixer-kernel 5 \
  --amp \
  --log-every 100 \
  2>&1 | tee runs/byte_dense.log
```

UABLA byte LM:

```bash
PYTHONPATH=src python scripts/run_byte_lm.py \
  --attention uabla \
  --text-file data/wiki_sample.txt \
  --seq-len 512 \
  --steps 5000 \
  --batch-size 8 \
  --hidden-size 128 \
  --layers 4 \
  --byte-mixer-kernel 5 \
  --byte-route-patch-size 16 \
  --amp \
  --log-every 100 \
  2>&1 | tee runs/byte_uabla.log
```

Primary metrics:

```text
loss
lm_loss
byte_accuracy
byte_perplexity
answer_accuracy for needle tasks
answer_loss for needle tasks
tokens_per_second
cache_dim_per_token_per_layer
route_entropy
avg_open_blocks
avg_token_budget
```

The first byte-LM goal is not to beat dense immediately. It is to measure
whether UABLA degrades more gracefully as raw-byte sequence length grows.

## Byte Needle Recall

Plain byte LM is mostly local at 512 bytes. To test long-range routing, use
the `needle` task. It injects a random alphanumeric byte code:

```text
SECRET_CODE: A7K2...
... long story gap ...
QUESTION: What is the SECRET_CODE?
ANSWER: A7K2...
```

Answer metrics are measured only on the answer code bytes. Local attention
should fail when `--needle-min-gap` is larger than `--local-window`.

Dense:

```bash
PYTHONPATH=src python scripts/run_byte_lm.py \
  --task needle \
  --attention dense \
  --text-file data/tinystories_combo.txt \
  --seq-len 1024 \
  --needle-min-gap 768 \
  --needle-code-length 12 \
  --steps 5000 \
  --batch-size 8 \
  --hidden-size 128 \
  --layers 4 \
  --byte-mixer-kernel 5 \
  --lm-loss-weight 0.2 \
  --answer-loss-weight 5.0 \
  --amp \
  --log-every 100 \
  2>&1 | tee runs/byte_needle_dense.log
```

Local:

```bash
PYTHONPATH=src python scripts/run_byte_lm.py \
  --task needle \
  --attention local \
  --text-file data/tinystories_combo.txt \
  --seq-len 1024 \
  --needle-min-gap 768 \
  --needle-code-length 12 \
  --local-window 128 \
  --steps 5000 \
  --batch-size 8 \
  --hidden-size 128 \
  --layers 4 \
  --byte-mixer-kernel 5 \
  --lm-loss-weight 0.2 \
  --answer-loss-weight 5.0 \
  --amp \
  --log-every 100 \
  2>&1 | tee runs/byte_needle_local.log
```

UABLA with memory-lite diagnostics:

```bash
PYTHONPATH=src python scripts/run_byte_lm.py \
  --task needle \
  --attention uabla \
  --text-file data/tinystories_combo.txt \
  --seq-len 1024 \
  --needle-min-gap 768 \
  --needle-code-length 12 \
  --local-window 128 \
  --steps 5000 \
  --batch-size 8 \
  --hidden-size 128 \
  --layers 4 \
  --byte-mixer-kernel 5 \
  --byte-route-patch-size 16 \
  --routed-span-left 2 \
  --routed-span-right 8 \
  --lm-loss-weight 0.2 \
  --answer-loss-weight 5.0 \
  --diagnostics-every 500 \
  --amp \
  --log-every 100 \
  2>&1 | tee runs/byte_needle_uabla.log
```
