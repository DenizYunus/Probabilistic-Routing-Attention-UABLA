"""UABLA-448 prototype components."""

from .config import UABLAConfig

__all__ = ["UABLAAttention", "UABLAConfig", "UABLAOutput"]


def __getattr__(name: str):
    if name in {"UABLAAttention", "UABLAOutput"}:
        from .attention import UABLAAttention, UABLAOutput

        return {"UABLAAttention": UABLAAttention, "UABLAOutput": UABLAOutput}[name]
    raise AttributeError(name)
