# Configuration and resume

## Paths

Paths in public configs resolve against the **repository root**, independently
of the shell's working directory. Absolute paths and environment variables are
also supported. To reuse models and data on another disk, create an ignored
`configs/machine.local.yaml`:

```yaml
_base_: chunkwise_4step.yaml
paths:
  wan: /your/model/disk/Wan2.1-T2V-1.3B
  dit: /your/model/disk/Wan2.1-T2V-14B-Diffusers
  videomae: /your/model/disk/VideoMAEv2-Large
  taew: /your/model/disk/taew2_1.pth
  output: /your/output/disk/direct-forcing
```

Set `_base_: chunkwise_4step_v16.yaml` for the V16/V16 baseline. Omitted values
inherit from that config. For DiT, `paths.dit` is the directory **containing**
`transformer/`; for VideoMAE, `paths.videomae` contains `model.safetensors` and the
downloaded Python/config files. `paths.taew` points directly to the `.pth` file.

Use the same config throughout preparation and training:

```bash
python scripts/download_models.py --config configs/machine.local.yaml
python scripts/prepare_data.py --config configs/machine.local.yaml \
  --manifest /path/to/dataset.jsonl
bash scripts/prepare_references.sh configs/machine.local.yaml
python train.py --config configs/machine.local.yaml --prepare-only
torchrun --standalone --nproc_per_node=8 train.py --config configs/machine.local.yaml
```

The example manifest path is an input supplied by the user. Choose either
`--manifest` or `--source-lmdb` as explained in the README.

## Memory and GPU count

The default main recipe uses **CCA 32 / RDM 128**, with **250 updates**, on eight
GPUs. The V16/V16 baseline uses **CCA 32 / RDM 64**, with **150 updates**.
Four-GPU training preserves the selected global batches through gradient accumulation:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/chunkwise_4step.yaml --set training.world_size=4
```

This changes the work per GPU; it is not a claim of identical training speed.
The frozen features and references still need to fit the selected memory strategy.
The public defaults use:

```yaml
memory:
  reference_residency: cpu
  activation_offload: true
  cache_cca_targets: true
```

`reference_residency=cuda` keeps RDM reference matrices on GPU. The main config
has two large references, so switch only when device memory permits. Activation
offload controls saved generator activations separately. `cache_cca_targets`
caches frozen real-video CCA features; it does not cache generated features.

## Gradient balancing

GNM and REW are enabled by default. GNM matches the generator-gradient norms;
REW scales the calibrated CCA gradient using its residual fitting error. See
[the paper](https://arxiv.org/abs/2610.30861).

GNM with a fixed post-normalization CCA/RDM gradient ratio of 0.5:

```bash
bash scripts/train_chunkwise_4step_v16.sh \
  --set controller.rew=false controller.fixed_ratio=0.5
```

Fixed CCA loss weight with both modules disabled:

```bash
bash scripts/train_chunkwise_4step_v16.sh \
  --set controller.gnm=false controller.rew=false controller.fixed_weight=1.0
```

REW uses the GNM-calibrated gradient; a REW-only configuration is rejected.

## Checkpoints and resume

Every run has a new output directory. A checkpoint contains:

```text
outputs/chunkwise_4step/RUN/checkpoint_model_000250/
├── model.pt       # generator parameters and step; sufficient for inference
└── training.pt    # optimizer, controller, RNG state, and training contract
```

To continue a BS128 run from step 150 to 250, pass the **checkpoint directory**:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --config configs/chunkwise_4step.yaml \
  --resume outputs/chunkwise_4step/RUN/checkpoint_model_000150 \
  --set training.steps=250
```

Replace `RUN` with an actual run directory. Resume writes to a new run directory
and restores the optimizer/controller. The fixed cosine horizon remains at step
125, so later updates use the configured learning-rate floor. Reference, encoder,
dataset, and objective identities are checked before restoring the training state.

An earlier BS64 main-recipe checkpoint still requires `training.rdm_batch=64`
when resuming. The new default does not change the batch recorded in a saved run.
Legacy controller and reference metadata are translated at the loading boundary;
generator parameters and optimizer tensor keys are unchanged.

Inference takes the **model file** instead:

```bash
GPUS=1 bash scripts/infer_chunkwise_4step.sh \
  --checkpoint outputs/chunkwise_4step/RUN/checkpoint_model_000250/model.pt \
  --prompts assets/prompts/demo.txt --output outputs/demo
```

## W&B

W&B is disabled by default. After `wandb login`, enable it with:

```bash
bash scripts/train_chunkwise_4step.sh \
  --set logging.wandb_mode=online logging.project=YOUR_PROJECT
```

Online recording includes training metrics, prompts, and video previews. Use
`logging.wandb_mode=offline` for local W&B files. Authentication comes from the
login/environment, rather than a committed configuration file.
