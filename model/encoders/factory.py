"""Frozen VideoMAE v2 and DiT feature factory for the released recipes."""
DIT_ENCODER_IDS = ("wan21_14b_block8_bid_clean_tau300_pooled",)
ENCODER_IDS = ("videomae_v2_vitl_l15_5x16_224", "videomae_v2_vitl_l23_5x16_224", *DIT_ENCODER_IDS)

def build_video_encoder(config, device, *, activation_checkpointing=None):
    if config.get("rdm_representation") != "tokens":
        raise ValueError("The released short-video recipe uses token features")
    model_id = str(config.rdm_encoder_id)
    if model_id not in ENCODER_IDS:
        raise ValueError(f"Unsupported release encoder: {model_id}")
    checkpointing = bool(config.rdm_encoder_activation_checkpointing) if activation_checkpointing is None else activation_checkpointing
    if model_id in DIT_ENCODER_IDS:
        from model.encoders.dit import WanDiTFeatureEncoder
        encoder = WanDiTFeatureEncoder(config, device)
    else:
        from model.encoders.videomae_v2 import VideoMAEv2LargeVideoEncoder
        encoder = VideoMAEv2LargeVideoEncoder(model_id, config.rdm_encoder_checkpoint_path,
            checkpoint_key=str(config.rdm_encoder_checkpoint_key), input_height=480, input_width=832,
            input_frames=81, padded_frames=81, temporal_pad="skip_first", feature_dim=1024,
            activation_checkpointing=checkpointing, device=device)
    from model.encoders.tokens import TokenVideoEncoder
    return TokenVideoEncoder(encoder, int(config.rdm_token_count), config)
