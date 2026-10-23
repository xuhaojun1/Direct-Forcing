#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DIRECT_FORCING_PYTHON=${PYTHON:-python}
exec "$DIRECT_FORCING_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-1}" \
  inference.py "$@"
