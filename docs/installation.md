# Model assets

Download the pretrained assets to `checkpoints/`:

```bash
python scripts/download_models.py
```

For inference only, use `python scripts/download_models.py --models wan`.
Trained Direct-Forcing checkpoints are not distributed in this release.

| Model | Purpose | Required for | Default location |
|---|---|---|---|
| [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Generator initialization, Wan T5 text encoder/tokenizer, and original VAE | Training and inference | `checkpoints/Wan2.1-T2V-1.3B/` |
| [VideoMAE v2 Large](https://huggingface.co/OpenGVLab/VideoMAEv2-Large) | Frozen L16/L24 video features for CCA/RDM | Training | `checkpoints/VideoMAEv2-Large/` |
| [Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Frozen DiT L9 features for RDM | Training | `checkpoints/Wan2.1-T2V-14B-Diffusers/transformer/` |
| [TAEW2.1](https://github.com/madebyollin/taehv) | Lightweight differentiable decoder for training | Training | `checkpoints/taew2_1.pth` |
| [SigLIP2 SO400M](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-256) | Frozen text features for the RDM video-text kernel | Data preparation | Hugging Face cache; encoded table at `data/train/siglip2.npy` |

DiT uses the **Diffusers** `transformer/` layout, retaining the first nine blocks
for L9 features. VideoMAE L16 and L24 share one checkpoint. SigLIP2 is fetched
automatically during data preparation.

## Expected model files

```text
checkpoints/
├── Wan2.1-T2V-1.3B/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.1_VAE.pth
│   └── google/umt5-xxl/                 # tokenizer files
├── Wan2.1-T2V-14B-Diffusers/
│   └── transformer/
│       ├── config.json
│       ├── diffusion_pytorch_model.safetensors.index.json
│       └── diffusion_pytorch_model-00001-of-00012.safetensors  # all 12 shards
├── VideoMAEv2-Large/
│   ├── config.json
│   ├── modeling_config.py
│   ├── modeling_videomaev2.py
│   └── model.safetensors
└── taew2_1.pth
```

Approximate download sizes are 17.6 GB for Wan 1.3B, 57.2 GB for the full DiT
transformer shards, 1.22 GB for VideoMAE, and 22.7 MB for TAEW. Data, SigLIP2,
references, and training outputs use additional space.
