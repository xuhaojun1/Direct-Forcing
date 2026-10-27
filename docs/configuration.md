# Configuration

All paths resolve from the repository root. Place model assets as follows:

```text
checkpoints/
├── Wan2.1-T2V-1.3B/
├── VideoMAEv2-Large/
├── Wan2.1-T2V-14B-Diffusers/
│   └── transformer/
└── taew2_1.pth
```

See [expected model files](installation.md#expected-model-files) for each folder's contents.
The downloader uses these locations automatically.

## Data preparation

Download the [Causal-Forcing data](https://huggingface.co/zhuhz22/Causal-Forcing-data)
into `data/Causal-Forcing/`:

```bash
hf download zhuhz22/Causal-Forcing-data --local-dir data/Causal-Forcing
python scripts/prepare_data.py
```

Preparation reads the chunk-wise shards and writes training caches to `data/train/`.

## Training

```bash
bash scripts/prepare_references.sh
bash scripts/train_chunkwise_4step.sh
```

References are saved in `data/references/`; training outputs go to `outputs/chunkwise_4step/`.
Settings are defined in `configs/chunkwise_4step.yaml`.

## W&B

Logging is disabled by default. To enable it:

```bash
wandb login
bash scripts/train_chunkwise_4step.sh \
  --set logging.wandb_mode=online logging.project=YOUR_PROJECT
```
