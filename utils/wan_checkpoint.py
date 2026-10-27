"""Load one authoritative student checkpoint before FSDP2 takes local shards."""

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict


def restore_rdm_source(model, checkpoint_path, *, native_state=None):
    """Reset a RDM branch to model weights only, never a previous optimizer."""
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    weights = {}
    if rank == 0:
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            weights = state.get("model", state)
        elif native_state:
            weights = dict(native_state)
        else:
            raise ValueError("Native RDM initialization requires its saved base weights")
    if not distributed:
        return model.load_state_dict(weights, strict=True)
    return set_model_state_dict(model, weights, options=StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=True, strict=True))


def load_rank0_checkpoint(model, state_dict, device):
    """Rank zero supplies the full state; every rank receives the same weights.

    FSDP2 fully_shard() does NOT broadcast initial parameters. Loading only
    rank zero and then sharding mixes checkpoint rows with other ranks' base
    pretrained rows. Synchronize the full state before sharding or LoRA setup.
    Without a checkpoint, preserve the original pretrained initialization.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        result = model.load_state_dict(state_dict, strict=True) if state_dict is not None else None
        return model, result

    # All ranks take the same branch even though only rank zero owns the dict.
    present = torch.tensor(int(state_dict is not None), device=device, dtype=torch.int32)
    dist.broadcast(present, src=0)
    if not present.item():
        return model, None
    # NCCL communicates CUDA tensors. This transient full student allocation
    # precedes the frozen teacher/T5/VAE; fully_shard releases the full weights.
    model = model.to(device=device)
    result = set_model_state_dict(
        model, state_dict if dist.get_rank() == 0 else {},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True),
    )
    return model, result
