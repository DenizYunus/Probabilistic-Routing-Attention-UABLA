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
