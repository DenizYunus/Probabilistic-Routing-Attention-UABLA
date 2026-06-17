from __future__ import annotations

from argparse import Namespace

import pytest

pytest.importorskip("torch")

from scripts.run_byte_lm import UABLA_V1_BYTE_NEEDLE_RECIPE, apply_recipe


def test_none_recipe_leaves_args_untouched() -> None:
    args = Namespace(recipe="none", token_contrast_weight=0.123)

    apply_recipe(args)

    assert args.token_contrast_weight == 0.123


def test_uabla_v1_byte_needle_recipe_locks_validated_settings() -> None:
    args = Namespace(recipe=UABLA_V1_BYTE_NEEDLE_RECIPE)

    apply_recipe(args)

    assert args.attention == "uabla"
    assert args.task == "needle"
    assert args.local_window == 128
    assert args.byte_route_patch_size == 16
    assert args.routed_span_left == 2
    assert args.routed_span_right == 8
    assert args.route_budget_curriculum == "explore"
    assert args.route_budget_curriculum_boundaries == "2500,6500"
    assert args.stage_shifted_when_answer_accuracy == 0.5
    assert args.stage_shifted_min_step == 6501
    assert args.route_entropy_weight == 0.02
    assert args.route_entropy_min == 0.35
    assert args.route_entropy_decay_steps == 6500
    assert args.direct_route_weight == 0.0
    assert args.token_contrast_weight == 0.0005
    assert args.token_contrast_warmup_steps == 1000
    assert args.token_contrast_decay_steps == 3500
    assert args.token_contrast_temperature == 0.7
