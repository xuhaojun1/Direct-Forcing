# VBench evaluation

Generate videos in the Direct-Forcing environment and score them in a separate
VBench environment.

## Setup

Follow the [VBench installation instructions](https://github.com/Vchitect/VBench),
including evaluator weights and dimension dependencies, then run:

```bash
VBENCH_PYTHON=/path/to/vbench/python bash scripts/setup_vbench.sh
```

## Generate and score

Run from the Direct-Forcing environment, replacing `RUN` with your training run:

```bash
bash scripts/eval_vbench.sh \
  --checkpoint outputs/chunkwise_4step/RUN/checkpoint_model_000250/model.pt \
  --output outputs/vbench/main --gpus 8 \
  --vbench-python /path/to/vbench/python
```

Add `--generate-only` or `--score-only` to run one stage separately.

## Prompt protocol

Evaluation uses **944 unique prompts × 5 samples** (4,720 videos), 81 frames at
16 fps, and all **16 VBench dimensions**. Generation uses expanded prompts;
scoring uses the original VBench prompts.

## Results

- `outputs/vbench/main/scores.json`: summary scores.
- `outputs/vbench/main/scores/`: detailed results.

Summary scores are fractions; multiply by 100 for percentages.
Use a new output directory for each checkpoint.
