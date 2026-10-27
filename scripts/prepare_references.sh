#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DIRECT_FORCING_CONFIG=${1:-configs/chunkwise_4step.yaml}
if [[ $# -gt 0 ]]; then shift; fi
DIRECT_FORCING_PYTHON=${PYTHON:-python}
DIRECT_FORCING_GPUS=${GPUS:-8}
DIRECT_FORCING_ENCODER_LIST=$("$DIRECT_FORCING_PYTHON" -c 'import sys; from utils.config import load_spec; print("\n".join(load_spec(sys.argv[1]).features.rdm))' "$DIRECT_FORCING_CONFIG")
mapfile -t DIRECT_FORCING_ENCODERS <<< "$DIRECT_FORCING_ENCODER_LIST"
for DIRECT_FORCING_ENCODER in "${DIRECT_FORCING_ENCODERS[@]}"; do
  "$DIRECT_FORCING_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$DIRECT_FORCING_GPUS" \
    scripts/prepare_reference.py --config "$DIRECT_FORCING_CONFIG" --encoder "$DIRECT_FORCING_ENCODER" "$@"
done
