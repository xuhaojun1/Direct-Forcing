"""Frozen TAEW training decoder and its cache/checkpoint identity."""
from pathlib import Path
from references.bank import file_sha256
from model.decoders.taew import TAEW21FullDecoder


def training_decoder_contract(config):
    if config.get('rdm_training_decoder', 'taew') != 'taew':
        raise ValueError('The release training recipe uses TAEW')
    return dict(name='taew', checkpoint_sha256=str(config.rdm_taew_checkpoint_sha256),
        gradient='full_temporal_vjp', input='normalized_Wan21_latent_BTC_HW',
        output='pixels_BTC_HW_float_minus1_plus1', frozen=True)


def build_training_decoder(config, device, dtype):
    contract = training_decoder_contract(config)
    path = str(Path(config.rdm_taew_checkpoint_path).resolve())
    if file_sha256(path) != contract['checkpoint_sha256']:
        raise RuntimeError('TAEW training decoder checkpoint checksum mismatch')
    return TAEW21FullDecoder(checkpoint_path=path, device=device, dtype=dtype, parallel=False)
