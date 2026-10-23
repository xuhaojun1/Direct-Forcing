"""Initialization policy for the short-video release."""

def validate_initialization(config):
    if config.get("joint_initialization", "wan") != "wan":
        raise ValueError("This release trains from original Wan; use --resume for a Direct-Forcing checkpoint")
    if config.get("generator_ckpt") or config.get("resume_ckpt"):
        raise ValueError("Use the explicit Direct-Forcing resume option, not an implicit generator override")
