# Direct-Forcing

**A Single-Stage Training Recipe for Few-Step Autoregressive Text-to-Video Generation**

[![arXiv](https://img.shields.io/badge/arXiv-2610.30861-b31b1b.svg)](https://arxiv.org/abs/2610.30861)

[Paper](https://arxiv.org/abs/2610.30861) · [Samples](#samples) · [Overview](#overview) · [Installation](#installation) · [Model downloads](#model-downloads) · [Inference](#inference) · [Training](#training) · [VBench](#vbench)

Direct-Forcing jointly trains a causal Wan generator with **causal continuation
alignment (CCA)** and **rollout distribution matching (RDM)**. This release
provides the short-video, **4-step chunk-wise** workflow: data preparation,
reference construction, training, inference, and VBench evaluation.

## Samples

[![Four Direct-Forcing generation examples on the left and the training-pipeline teaser on the right](assets/figures/showcase.jpg)](assets/figures/showcase.jpg)

Left: four selected generations, each showing frames **0, 40, and 80** from our
150-update main model. Right: the paper teaser, including the **250-update** result.
[Full prompts](assets/samples/samples.json) · [Teaser PDF](assets/figures/teaser.pdf)

## Overview

[![Causal alignment, rollout distribution matching, and gradient balancing](assets/figures/pipeline.png)](assets/figures/pipeline.pdf)

Click the pipeline figure for its vector PDF. The default main recipe uses
RDM batch 128 and 250 updates, as reported in the [paper](https://arxiv.org/abs/2610.30861).

| Configuration | CCA encoder | RDM encoders | Global CCA / RDM batch | Updates |
|---|---|---|---|---|
| `configs/chunkwise_4step.yaml` (default) | VideoMAE v2 L24 | VideoMAE v2 L16 + DiT L9 | 32 / 128 | 250 |
| `configs/chunkwise_4step_v16.yaml` | VideoMAE v2 L16 | VideoMAE v2 L16 | 32 / 64 | 150 |

Both configurations use original Wan initialization, SigLIP2 text features,
TAEW for training, and **GNM + REW**. Each chunk contains
3 latent frames. Every chunk uses the four-step denoising grid at inference;
training retains the stochastic-exit self-rollout and staged VJP implementation.
There is no optical-flow loss or model EMA in these recipes.

## Installation

Use **Linux x86_64, Python 3.11, and an NVIDIA GPU with BF16 support**. The default
training launch uses **8 GPUs**, with reference and activation offload enabled
for the 96 GB configuration. Inference runs on a single GPU.

The commands below use the documented [training environment](docs/installation.md).
The NVIDIA driver must support CUDA 13.0.

```bash
git clone https://github.com/xuhaojun1/Direct-Forcing.git
cd Direct-Forcing

conda create -n direct-forcing python=3.11 -y
conda activate direct-forcing
python -m pip install --upgrade pip setuptools wheel

# Install the matching PyTorch / torchvision pair first.
python -m pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130

# Install the remaining dependencies and the local project.
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . --no-deps
python -m pip check
```

If FlashAttention needs to compile locally, install a CUDA 13.0 toolkit with
`nvcc` before the FlashAttention command. See
[CUDA setup, source builds, and troubleshooting](docs/installation.md).
The CUDA version shown by `nvidia-smi` is the driver's supported version; it does
not confirm that a compiler is installed.

Check the installation before downloading models:

```bash
python scripts/check_environment.py
```

This checks package imports, pinned feature-library versions, and the visible
GPU. `--imports-only` checks Python dependencies on a CPU/login node. An optional
`--attention-test` runs a small FlashAttention forward/backward on one GPU.
Keep VBench in a [separate environment](#vbench).

## Model downloads

### What each model is used for

Downloaded Wan, VideoMAE, and TAEW assets are placed under `checkpoints/`. The helper reads
[`configs/chunkwise_4step.yaml`](configs/chunkwise_4step.yaml), pins upstream
revisions, and records the selected revision beside each model.

| Model | Purpose | Required for | Default location |
|---|---|---|---|
| [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Generator initialization, Wan T5 text encoder/tokenizer, and original VAE | Training and inference | `checkpoints/Wan2.1-T2V-1.3B/` |
| [VideoMAE v2 Large](https://huggingface.co/OpenGVLab/VideoMAEv2-Large) | Frozen L16/L24 video features for CCA/RDM | Main and V16 baseline training | `checkpoints/VideoMAEv2-Large/` |
| [Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Frozen DiT L9 features for RDM | Main configuration training | `checkpoints/Wan2.1-T2V-14B-Diffusers/transformer/` |
| [TAEW2.1](https://github.com/madebyollin/taehv) | Lightweight differentiable decoder for training | Main and V16 baseline training | `checkpoints/taew2_1.pth` |
| [SigLIP2 SO400M](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-256) | Frozen text features for the RDM video-text kernel | Data preparation | Hugging Face cache; encoded table at `data/train/siglip2.npy` |

**The main recipe uses the 14B Diffusers repository above.** Its `transformer/`
layout is required by our feature loader. The helper downloads that subfolder;
the feature loader retains the first nine blocks in memory for DiT L9. VideoMAE L16 and L24 share
one downloaded VideoMAE checkpoint.

Wan's **T5** supplies text conditioning to the generator and DiT feature encoder.
**SigLIP2** supplies a separate, 1152-dimensional text representation to the RDM
kernel. OpenCLIP downloads SigLIP2 automatically on the first data-preparation run;
subsequent training reads the encoded text table.

### Download for your task

**Main configuration: CCA=V24, RDM=V16+D9**

```bash
# Optional: inspect sources and destinations without downloading.
python scripts/download_models.py --list

# Download Wan 1.3B, VideoMAE v2, DiT 14B, and TAEW.
python scripts/download_models.py
```

**V16 baseline: CCA=RDM=V16**

```bash
python scripts/download_models.py --config configs/chunkwise_4step_v16.yaml
```

**Inference only**

```bash
python scripts/download_models.py --models wan
```

Inference also requires a **trained Direct-Forcing checkpoint**. The downloader
obtains upstream model assets; this source release does not yet provide a public
Direct-Forcing checkpoint download. Use `model.pt` from a completed training run
with the [inference command below](#inference).

<details>
<summary>Expected model directory after downloading the main configuration</summary>

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

</details>

<details>
<summary>Existing model directories, Hugging Face cache, proxy, and interrupted downloads</summary>

To reuse existing models, create `configs/machine.local.yaml` following the
[path configuration guide](docs/configuration.md#paths), and use it consistently:

```bash
python scripts/download_models.py --config configs/machine.local.yaml --list
python scripts/download_models.py --config configs/machine.local.yaml
```

Download individual assets with `--models wan`, `--models videomae`,
`--models dit`, or `--models taew`. Hugging Face downloads reuse completed files
when the same command is rerun. An interrupted TAEW download restarts that small
file and verifies its SHA-256 before installing it.

To keep automatically fetched models in this checkout's ignored cache:

```bash
export HF_HOME="$PWD/cache/huggingface"
```

If your network requires a proxy, set `http_proxy` and `https_proxy` in the shell
before cloning or downloading. Use your own proxy address, for example:

```bash
export http_proxy="http://PROXY_HOST:PORT"
export https_proxy="$http_proxy"
```

The upstream models listed here are public. Use `hf auth login` only if your
Hugging Face access requires authentication; credentials stay outside configs.

</details>

## Inference

After downloading Wan 1.3B, point to a trained `model.pt` containing the
`generator` state. Run from the repository root:

```bash
GPUS=1 bash scripts/infer_chunkwise_4step.sh \
  --checkpoint outputs/chunkwise_4step/RUN/checkpoint_model_000250/model.pt \
  --prompts assets/prompts/demo.txt --output outputs/demo
```

This produces 81-frame videos at 16 fps using four denoising steps for every
3-latent-frame chunk. Multiple GPUs split prompts; they do not require model
parallelism. `--decode-offload` moves the generator off GPU before full VAE decoding.

## Training

### 1. Prepare data

Training data and precomputed caches are **not distributed**. Supply a dataset
you have permission to use. The default workflow retains 81 RGB frames at 16 fps,
480×832 resolution, corresponding to 21 normalized Wan latent frames.

For an **already merged clean-latent LMDB** in the
[Causal-Forcing format](https://huggingface.co/zhuhz22/Causal-Forcing-data):

```bash
python scripts/prepare_data.py --source-lmdb /path/to/causal-forcing/clean_data
```

The input directory must contain `data.mdb` with clean latent rows; a Hugging
Face download containing multiple shards must be merged first using the upstream
Causal-Forcing data-preparation tools. The import selects the final clean latent
from each row without decoding and re-encoding it or applying Wan normalization
twice. It creates a separate dataset and does
not modify the source LMDB.

For raw videos, create a JSONL manifest following `assets/dataset.example.jsonl`:

```json
{"video": "videos/00001.mp4", "prompt": "A small sailboat glides across a calm lake.", "start_seconds": 0.0}
```

```bash
python scripts/prepare_data.py --manifest /path/to/dataset.jsonl
```

Video paths are resolved relative to the manifest. Videos are sampled at 16 fps,
resized to cover 480×832, and center-cropped. The default interval is the first
five seconds; `start_seconds` selects another five-second interval. Source videos
must cover that interval. Use at least 288 rows for the main recipe, or 224 for
the V16 baseline: the fixed diagnostic split has 160 rows, excluded from
optimizer sampling, leaving at least one complete RDM batch.

The preparation script produces clean-latent and prompt LMDBs, a normalized
1152-dimensional SigLIP2 text table, fixed probe indices, and dataset hashes.
SigLIP2 retains its upstream text-token limit; prompts are not automatically
rewritten or shortened.

### 2. Build RDM references

```bash
bash scripts/prepare_references.sh configs/chunkwise_4step.yaml
```

For the baseline:

```bash
bash scripts/prepare_references.sh configs/chunkwise_4step_v16.yaml
```

Each encoder extracts features across 8 workers, then rank 0 builds its Nyström
reference. Set `GPUS=4` to use four extraction workers. Extraction is resumable;
row contracts prevent incompatible caches from being mixed. VideoMAE references
use the original Wan VAE, and DiT references use clean normalized latents with
Wan text conditioning. Training uses TAEW. Each RDM reference has its own kernel
bandwidth and SigLIP2 scaling. CCA target features are cached lazily during training.

### 3. Train on 8 GPUs

Validate assets and configuration before allocating the generator:

```bash
python train.py --config configs/chunkwise_4step.yaml --prepare-only
```

Default paper configuration, 8 GPUs:

```bash
bash scripts/train_chunkwise_4step.sh
```

V16/V16 baseline, 8 GPUs:

```bash
bash scripts/train_chunkwise_4step_v16.sh
```

Both launchers enable **GNM + REW** with global CCA batch 32. The main recipe runs
250 updates with RDM batch 128; the V16 baseline runs 150 updates with RDM batch
64. For custom paths, 4-GPU execution, controller ablations, and
checkpoint resume, see [configuration and resume](docs/configuration.md).

Each run creates an isolated output directory with the public configuration,
resolved runtime configuration, JSONL metrics, previews, and checkpoints every
25 updates. Each checkpoint retains model, optimizer, controller, and RNG state.
The learning rate follows the existing cosine schedule through
update 125, then remains at its configured floor.

W&B is disabled by default. After `wandb login`, use
`--set logging.wandb_mode=online logging.project=YOUR_PROJECT`.

## VBench

Install VBench in a separate environment following its upstream dependency and
model-weight instructions. `scripts/setup_vbench.sh` clones the official repository
and performs an editable install; some dimensions additionally require native
extensions such as Detectron2. Set `VBENCH_REVISION` to a specific commit to pin it.

```bash
VBENCH_PYTHON=/path/to/vbench/python bash scripts/setup_vbench.sh
bash scripts/eval_vbench.sh \
  --checkpoint outputs/chunkwise_4step/RUN/checkpoint_model_000250/model.pt \
  --output outputs/vbench/main --gpus 8 \
  --vbench-python /path/to/vbench/python
```

The protocol preserves the existing experiments: 946 aligned prompt rows reduce
to **944 unique prompts**, using the corresponding expanded prompts for generation
and five samples per prompt (**4,720 videos**). The official metadata retains its
dimension annotations. `eval/score.py` computes all 16 dimensions;
`outputs/vbench/main/scores.json` contains Total, Quality, Semantic, Dynamic and
the raw dimensions. Summary values are fractions; multiply by 100 for percentages.

Use `--generate-only` or `--score-only` to separate environments or retry scoring.
`--prepare-only` validates the prompt protocol without generating videos. Output
manifests prevent evaluation folders from mixing different checkpoints.

## Citation

If you use Direct Forcing in your research, please cite our paper:

```bibtex
@misc{xu2026directforcing,
  title={Direct Forcing: A Single-Stage Training Recipe for Few-Step Autoregressive Text-to-Video Generation},
  author={Haojun Xu and Jie Huang and Xin Lu and Mingchen Zhong and Zihao Fan and Linjiang Huang and Si Liu},
  year={2026},
  eprint={2610.30861},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2610.30861}
}
```
