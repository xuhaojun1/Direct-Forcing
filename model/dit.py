"""Direct-Forcing adapter for the bidirectional WAN transformer.

The implementation remains Diffusers' WAN transformer, but keeping the adapter
inside :mod:`model.dit` gives the WAN training configs one stable local
model entry point and supplies the FSDP block policy used by Direct-Forcing.
"""

from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanTransformerBlock


class WanTransformer3DModel(DiffusersWanTransformer3DModel):
    _fsdp_shard_conditions = [
        lambda name, module: isinstance(module, WanTransformerBlock)
    ]


__all__ = ["WanTransformer3DModel"]
