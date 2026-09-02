#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TVM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${TVM_ROOT}/.venv/bin/python"
ONNX_PATH="./model/model.onnx"
INPUT_DTYPE="float32"
TARGET_DTYPE="custom[posites2]8"
OUTPUT="./model/GPT2_custom_posites2_8.so"
NUM_CORES=28
USE_VECTORIZE=false

export PYTHONPATH="${TVM_ROOT}/python:${TVM_ROOT}/.local/python:${SCRIPT_DIR}/.."
export TVM_LIBRARY_PATH="${TVM_ROOT}/build"
export LD_LIBRARY_PATH="${TVM_ROOT}/build:${TVM_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$(dirname -- "${OUTPUT}")"

"${PYTHON_BIN}" Compile.py \
    --onnx-path "${ONNX_PATH}" \
    --target "{\"kind\":\"llvm\",\"num-cores\":${NUM_CORES}}" \
    --input-dtype "${INPUT_DTYPE}" \
    --target-dtype "${TARGET_DTYPE}" \
    --use-vectorize "${USE_VECTORIZE}" \
    --output "${OUTPUT}"
