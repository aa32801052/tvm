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
input_dtype="${INPUT_DTYPE:-float32}"
target_dtype="custom[${type_name}]${bits}"

"${PYTHON_BIN}" Compile.py \
    --onnx-model-path ./model/imagenet100_resnet18.onnx \
    --input-dtype "${input_dtype}" \
    --target-dtype "${target_dtype}" \
    --use-optimization none \
    --work-dir "./tuning_logs_mobilenet/${type_name}_${bits}" \
    --max-trials-global 1000 \
    --model-dir ./model \
    --model-name "mobilenet_imagenet100_${type_name}_${bits}.so"
