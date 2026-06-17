# Probabilistic Routing Attention

This repository contains a clean V1 prototype of **UABLA-448**:

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
