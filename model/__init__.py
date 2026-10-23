"""Shared causal video models; heavyweight backbones load on demand."""

__all__ = ["CausalVideoModel"]


def __getattr__(name):
    if name == "CausalVideoModel":
        from model.causal_video import CausalVideoModel
        return CausalVideoModel
    raise AttributeError(name)
