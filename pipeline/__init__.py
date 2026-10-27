"""Generation and replay pipelines."""

__all__ = ["CausalInferencePipeline", "SelfRolloutPipeline"]


def __getattr__(name):
    if name == "CausalInferencePipeline":
        from pipeline.causal_inference import CausalInferencePipeline
        return CausalInferencePipeline
    if name == "SelfRolloutPipeline":
        from pipeline.self_rollout import SelfRolloutPipeline
        return SelfRolloutPipeline
    raise AttributeError(name)
