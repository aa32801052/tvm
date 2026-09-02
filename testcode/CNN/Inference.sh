#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TVM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${TVM_ROOT}/python:${TVM_ROOT}/.local/python"
export TVM_LIBRARY_PATH="${TVM_ROOT}/build"

if [[ -x "${TVM_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${TVM_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

type_name="${TYPE_NAME:-posites2}"
bits="${BITS:-32}"
dtype="custom[${type_name}]${bits}"
model="${MODEL:-mobilenet_imagenet100_${type_name}_${bits}.so}"

"${PYTHON_BIN}" Inference.py \
    --dtype "${dtype}" \
    --data-root ./imagenet100_hf/validation \
    --model-path "./model/${model}" \
    --results_dir "./mobilenet_eval_results/${type_name}_${bits}" \
    --start_idx 1 \
    --end_idx 2
