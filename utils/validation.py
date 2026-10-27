"""Supported short-video runtime; retired research branches fail before loading."""


def validate_short_video_runtime(config):
    required = dict(rdm_generation_mode='chunkwise', num_frame_per_block=3,
        num_training_frames=21, rdm_context_gradient_mode='frozen',
        rdm_training_decoder='taew', rdm_representation='tokens',
        joint_cca_history_policy='matched', joint_cca_objective='feature',
        rdm_dynamic_reg_enabled=False, independent_first_frame=False,
        same_step_across_blocks=True, rdm_all_chunks_random_x0_4step=True,
        rdm_bidirectional_random_x0_4step=False, i2v=False,
        ts_schedule=False, ts_schedule_max=False)
    for key, expected in required.items():
        if config.get(key, expected) != expected:
            raise ValueError(f'Short-video Direct-Forcing requires {key}={expected!r}')
    for key in ('joint_rolling', 'joint_long_video_previews', 'resume_sf30_pooled',
                'resume_recover_sgf_writer', 'resume_recover_sgf_plus'):
        if config.get(key):
            raise ValueError(f'{key} is outside the short-video release')
    if list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError('Direct-Forcing uses four denoising steps for every chunk')
    if list(config.get('rdm_first_frame_denoising_step_list', [])):
        raise ValueError('The first chunk uses the same four-step schedule')
    if config.get('context_noise', 0) not in (0, 100, 357):
        raise ValueError('Unsupported context-noise replay level')
    model = config.get('model_kwargs', {})
    if model.get('sink_fifo_attention', False) or model.get('dual_full_expert'):
        raise ValueError('The short-video generator uses one set of weights and full causal history')
    if model.get('local_attn_size', -1) != -1 or model.get('sink_size', 0) != 0:
        raise ValueError('The short-video generator retains its full causal history')
