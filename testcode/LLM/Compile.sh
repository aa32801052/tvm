#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TVM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${TVM_ROOT}/.venv/bin/python"
ONNX_PATH="./model/model.onnx"
INPUT_DTYPE="float32"
TARGET_DTYPE="custom[posites1]32"
OUTPUT="./model/GPT2_custom_posites1_mixed_quire_32_8.so"
NUM_CORES=16
USE_VECTORIZE=false
USE_MIXED_PRECISION=true
MIXED_PRECISION_DTYPE="custom[posites1]8"
MIXED_PRECISION_ACC_DTYPE="custom[posites1]32"
USE_QUIRE=true

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
    --use-mixed-precision "${USE_MIXED_PRECISION}" \
    --mixed-precision-dtype "${MIXED_PRECISION_DTYPE}" \
    --mixed-precision-acc-dtype "${MIXED_PRECISION_ACC_DTYPE}" \
    --use-quire "${USE_QUIRE}" \
    --output "${OUTPUT}"
