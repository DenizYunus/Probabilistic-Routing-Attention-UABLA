from __future__ import annotations

import pytest

from uabla import UABLAConfig


def test_default_cache_dim_is_locked_to_448() -> None:
    config = UABLAConfig(hidden_size=7168)

    assert config.centroids_per_block == 4
    assert config.cache_dim == 448


def test_v1_rejects_non_four_centroids() -> None:
    with pytest.raises(ValueError, match="exactly 4 centroids"):
        UABLAConfig(hidden_size=128, centroids_per_block=3)
