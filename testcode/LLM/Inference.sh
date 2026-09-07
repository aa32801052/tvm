#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TVM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${TVM_ROOT}/.venv/bin/python"
MODEL1="./model/GPT2_fp32.so"
MODEL2="./model/GPT2_custom_posites1_mixed_32_8.so"
DTYPE1="float32"
DTYPE2="custom[posites1]32"
THREADS=28
ITER=32
PROMPT="Tell me about AI"
TEACHER_TOKENS="13,198,198,20185,318,257,649,2214,286,2267,326,468,587,1088,329,257,890,640,13,632,318,257,2214,326,468,587,1088,329,257,890,640,11"
DECODE_MODE="teacher"
TAG="fp32_vs_posit8es1_mixed"

export PYTHONPATH="${TVM_ROOT}/python:${TVM_ROOT}/.local/python:${SCRIPT_DIR}"
export TVM_LIBRARY_PATH="${TVM_ROOT}/build"
export LD_LIBRARY_PATH="${TVM_ROOT}/build:${TVM_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON_BIN}" Inference.py \
    --tag "${TAG}" \
    --model1 "${MODEL1}" \
    --model2 "${MODEL2}" \
    --dtype1 "${DTYPE1}" \
    --dtype2 "${DTYPE2}" \
    --num-threads "${THREADS}" \
    --iter "${ITER}" \
    --prompt "${PROMPT}" \
    --decode-mode "${DECODE_MODE}" \
    --teacher-tokens "${TEACHER_TOKENS}" \
