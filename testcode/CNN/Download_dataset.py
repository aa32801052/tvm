from datasets import load_dataset
from pathlib import Path

for split in ["train", "validation"]:
    ds = load_dataset("clane9/imagenet-100", split=split)
    out_root = Path("./imagenet100_hf") / split
    label_names = ds.features["label"].names

    for i, sample in enumerate(ds):
        img = sample["image"]
        label = sample["label"]
        class_dir = out_root / f"{label:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        img.save(class_dir / f"{i:06d}.jpg")