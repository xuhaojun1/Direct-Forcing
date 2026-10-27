#!/usr/bin/env bash
# Run inside a separate VBench environment, following its upstream CUDA guidance.
set -euo pipefail
cd "$(dirname "$0")/.."
DIRECT_FORCING_PYTHON=${VBENCH_PYTHON:-python}
if [[ ! -d third_party/VBench/.git ]]; then
  git clone https://github.com/Vchitect/VBench.git third_party/VBench
fi
if [[ -n "${VBENCH_REVISION:-}" ]]; then
  git -C third_party/VBench checkout --detach "$VBENCH_REVISION"
fi
"$DIRECT_FORCING_PYTHON" -m pip install -e third_party/VBench
git -C third_party/VBench rev-parse HEAD > third_party/VBench/direct_forcing_revision.txt
printf 'VBench installed. Follow its README to install optional dimension dependencies and download evaluator weights.\n'
