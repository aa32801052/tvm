from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="onnx-community/gpt2-ONNX",
    local_dir="./gpt2-ONNX"
)