import argparse
import json
import time
from pathlib import Path

import numpy as np
import tvm
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from custom_dtype_runtime import (
    configure_custom_datatypes,
    convert_custom_tensor_to_float_numpy,
    convert_float_array_to_custom_tensor,
    is_custom_dtype,
    normalize_dtype_suffix,
    validate_dtype_arg,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET = "llvm"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run TVM .so model evaluation on ImageNet-100")
    parser.add_argument(
        "--data-root",
        "--img_dir",
        dest="data_root",
        type=str,
        required=True,
        help="Path to ImageFolder-style validation directory",
    )
    parser.add_argument(
        "--imagenet-class-index",
        "--imagenet_class_index",
        dest="imagenet_class_index",
        type=str,
        default=None,
        help="Official ImageNet class index JSON for synset->ImageNet(1000) label mapping",
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=str,
        required=True,
        help="Path to compiled TVM model (.so)",
    )
    parser.add_argument(
        "--dtype",
        type=validate_dtype_arg,
        default="float32",
        help="Model dtype: float16/float32/custom[name]bits",
    )
    parser.add_argument("--start_idx", type=int, default=1, help="Start index for batch processing (1-based)")
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="End index for batch processing (1-based, -1 means all)",
    )
    parser.add_argument("--results_dir", type=str, default="./eval_results", help="Directory to save results")
    parser.add_argument("--merge", action="store_true", help="Merge all batch results and compute final accuracy")
    parser.add_argument("--input-size", type=int, default=224, help="Model input spatial size after crop")
    parser.add_argument("--resize-size", type=int, default=256, help="Resize shorter side before center crop")
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[float(x) for x in IMAGENET_MEAN],
        help="Normalization mean",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[float(x) for x in IMAGENET_STD],
        help="Normalization std",
    )
    parser.add_argument("--target", type=str, default=TARGET, help="TVM runtime target string")
    return parser.parse_args()


def build_transform(resize_size, input_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_imagenet_class_index(class_index_path):
    with open(class_index_path, "r") as f:
        class_index = json.load(f)

    synset_to_idx = {}
    idx_to_synset = {}
    idx_to_name = {}

    for k, v in class_index.items():
        idx = int(k)
        synset, class_name = v[0], v[1]
        synset_to_idx[synset] = idx
        idx_to_synset[idx] = synset
        idx_to_name[idx] = class_name

    return synset_to_idx, idx_to_synset, idx_to_name


def build_mapped_labels_from_dataset(dataset, synset_to_idx):
    local_to_imagenet = {}
    missing = []

    for local_idx, synset in enumerate(dataset.classes):
        if synset in synset_to_idx:
            local_to_imagenet[local_idx] = synset_to_idx[synset]
        else:
            missing.append(synset)

    if missing:
        preview = sorted(missing)[:20]
        raise RuntimeError(
            "Some class folders are not found in official class index. "
            f"Missing count={len(missing)}. Examples={preview}"
        )

    mapped_labels = [local_to_imagenet[local_idx] for _, local_idx in dataset.samples]
    return mapped_labels, local_to_imagenet


def load_existing_results(results_dir, start_idx, end_idx, dtype):
    results_dir = Path(results_dir)
    dtype_suffix = normalize_dtype_suffix(dtype)
    all_results = []

    batch_start = ((start_idx - 1) // 100) * 100 + 1
    batch_end = ((end_idx - 1) // 100 + 1) * 100 + 1

    for idx in range(batch_start, batch_end, 100):
        result_file = results_dir / f"results_{dtype_suffix}_{idx:06d}_{idx + 99:06d}.json"
        if not result_file.exists():
            continue

        try:
            with open(result_file, "r") as f:
                batch_data = json.load(f)
        except (json.JSONDecodeError, OSError, KeyError) as err:
            print(f"[WARN] Could not load {result_file}: {err}")
            continue

        if batch_data.get("dtype") != dtype:
            print(
                "[WARN] Skipping "
                f"{result_file} because dtype mismatch ({batch_data.get('dtype')} vs {dtype})"
            )
            continue

        for item in batch_data.get("results", []):
            if start_idx <= item["image_idx"] <= end_idx:
                all_results.append(item)

    return all_results


def save_single_result(results_dir, result, dtype, optimization):
    results_dir = Path(results_dir)
    dtype_suffix = normalize_dtype_suffix(dtype)

    image_idx = result["image_idx"]
    batch_start = ((image_idx - 1) // 100) * 100 + 1
    batch_end = batch_start + 99
    result_file = results_dir / f"results_{dtype_suffix}_{batch_start:06d}_{batch_end:06d}.json"

    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                batch_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            batch_data = {
                "dtype": dtype,
                "optimization": optimization if optimization else "None",
                "start_idx": batch_start,
                "end_idx": batch_end,
                "results": [],
            }
    else:
        batch_data = {
            "dtype": dtype,
            "optimization": optimization if optimization else "None",
            "start_idx": batch_start,
            "end_idx": batch_end,
            "results": [],
        }

    existing_idx = None
    for i, item in enumerate(batch_data["results"]):
        if item["image_idx"] == image_idx:
            existing_idx = i
            break

    if existing_idx is not None:
        batch_data["results"][existing_idx] = result
    else:
        batch_data["results"].append(result)

    batch_data["results"].sort(key=lambda x: x["image_idx"])

    batch_data["total_images"] = len(batch_data["results"])
    batch_data["correct_top1"] = sum(1 for x in batch_data["results"] if x["correct_top1"])
    batch_data["correct_top5"] = sum(1 for x in batch_data["results"] if x["correct_top5"])
    total_time = sum(x["inference_time_ms"] for x in batch_data["results"]) / 1000.0
    batch_data["total_time_seconds"] = total_time

    if batch_data["total_images"] > 0:
        batch_data["top1_accuracy"] = 100.0 * batch_data["correct_top1"] / batch_data["total_images"]
        batch_data["top5_accuracy"] = 100.0 * batch_data["correct_top5"] / batch_data["total_images"]
        batch_data["avg_inference_time_ms"] = total_time / batch_data["total_images"] * 1000.0
    else:
        batch_data["top1_accuracy"] = 0.0
        batch_data["top5_accuracy"] = 0.0
        batch_data["avg_inference_time_ms"] = 0.0

    temp_file = result_file.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(batch_data, f, indent=2)
    temp_file.replace(result_file)


def merge_results(results_dir, dtype=None):
    results_dir = Path(results_dir)

    if dtype:
        dtype_suffix = normalize_dtype_suffix(dtype)
        result_files = sorted(results_dir.glob(f"results_{dtype_suffix}_*.json"))
    else:
        result_files = sorted(results_dir.glob("results_*.json"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result files")
    print("Merging results...")

    all_results = []
    total_images = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_time = 0.0

    for result_file in tqdm(result_files, desc="Loading results"):
        with open(result_file, "r") as f:
            batch_data = json.load(f)

        total_images += batch_data["total_images"]
        total_correct_top1 += batch_data["correct_top1"]
        total_correct_top5 += batch_data["correct_top5"]
        total_time += batch_data["total_time_seconds"]
        all_results.extend(batch_data["results"])

    all_results.sort(key=lambda x: x["image_idx"])

    top1_accuracy = 100.0 * total_correct_top1 / total_images
    top5_accuracy = 100.0 * total_correct_top5 / total_images
    avg_time = total_time / total_images * 1000.0

    if dtype:
        dtype_suffix = normalize_dtype_suffix(dtype)
        merged_file = results_dir / f"merged_results_{dtype_suffix}.json"
    else:
        merged_file = results_dir / "merged_results.json"

    merged_data = {
        "total_images": total_images,
        "total_correct_top1": total_correct_top1,
        "total_correct_top5": total_correct_top5,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "total_time_seconds": total_time,
        "avg_inference_time_ms": avg_time,
        "num_batches": len(result_files),
        "results": all_results,
    }

    with open(merged_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    print("\n" + "=" * 80)
    print("Final Merged Results")
    print("=" * 80)
    print(f"Total Batches: {len(result_files)}")
    print(f"Total Images: {total_images}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}%")
    print(f"Average Inference Time: {avg_time:.2f} ms/image")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"\nMerged results saved to: {merged_file}")
    print("=" * 80)


def extract_logits(output, dtype, dev):
    first_output = output[0] if isinstance(output, (tuple, list)) else output

    if is_custom_dtype(dtype):
        logits = convert_custom_tensor_to_float_numpy(first_output, dtype, dev)
    else:
        logits = first_output.numpy()

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [N, C], but got shape {logits.shape}")
    return logits


def prepare_input_tensor(img_np, dtype, dev):
    if is_custom_dtype(dtype):
        return convert_float_array_to_custom_tensor(img_np, dtype, dev)
    return tvm.runtime.tensor(img_np, dev)


def maybe_register_custom_datatypes(dtype, target):
    if is_custom_dtype(dtype):
        target_kind = tvm.target.Target(target).kind.name
        print(f"Registering {dtype}...")
        configure_custom_datatypes([dtype], target=target_kind)


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    model_path = Path(args.model_path)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_results(results_dir, dtype=args.dtype)
        return

    maybe_register_custom_datatypes(args.dtype, args.target)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    transform = build_transform(
        resize_size=args.resize_size,
        input_size=args.input_size,
        mean=args.mean,
        std=args.std,
    )

    dataset = datasets.ImageFolder(root=str(data_root))
    samples = dataset.samples

    if not samples:
        raise RuntimeError(f"No images found under data root: {data_root}")

    print("=" * 80)
    print("Dataset Info")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Num classes: {len(dataset.classes)}")
    print(f"Num samples: {len(samples)}")
    print(f"First classes: {dataset.classes[:10]}")

    mapped_labels = None
    if args.imagenet_class_index:
        class_index_path = Path(args.imagenet_class_index)
        if not class_index_path.exists():
            raise FileNotFoundError(f"ImageNet class index not found: {class_index_path}")

        synset_to_idx, _, _ = load_imagenet_class_index(class_index_path)
        mapped_labels, local_to_imagenet = build_mapped_labels_from_dataset(dataset, synset_to_idx)

        print("\n" + "=" * 80)
        print("Label Mapping")
        print("=" * 80)
        print("Using folder synset -> official ImageNet index mapping")
        preview = []
        for local_idx, synset in enumerate(dataset.classes[:10]):
            preview.append((synset, local_to_imagenet[local_idx]))
        print(f"Mapping preview (first 10): {preview}")
    else:
        print("\n[INFO] No --imagenet-class-index provided; using ImageFolder local class indices")

    total_dataset_images = len(samples)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else total_dataset_images

    if start_idx < 1:
        raise ValueError("start_idx must be >= 1")
    if end_idx > total_dataset_images:
        raise ValueError(f"end_idx ({end_idx}) exceeds total images ({total_dataset_images})")
    if start_idx > end_idx:
        raise ValueError(f"start_idx ({start_idx}) must be <= end_idx ({end_idx})")

    array_start = start_idx - 1
    array_end = end_idx

    print(f"\nProcessing images {start_idx} to {end_idx} ({array_end - array_start} images)")

    existing_results = load_existing_results(results_dir, start_idx, end_idx, args.dtype)
    already_processed = set(item["image_idx"] for item in existing_results)
    print(f"Found {len(already_processed)} already processed images (dtype={args.dtype}), will skip them")

    print(f"\nLoading compiled model from {model_path}...")
    dev = tvm.device_from_target(tvm.target.Target(args.target), 0)
    lib = tvm.runtime.load_module(str(model_path))
    vm = tvm.relax.VirtualMachine(lib, dev)

    print(f"\nRunning inference with dtype={args.dtype}...")

    total = 0
    correct1 = 0
    correct5 = 0
    total_time = 0.0
    skipped = 0
    failed = 0
    label_mode = None
    num_model_classes = None

    for sample_idx in tqdm(range(array_start, array_end), desc="Evaluating", unit="image"):
        image_idx = sample_idx + 1

        if image_idx in already_processed:
            skipped += 1
            continue

        image_path, local_label = samples[sample_idx]

        try:
            with Image.open(image_path) as img:
                img_np = transform(img.convert("RGB")).numpy().astype(np.float32, copy=False)
            img_np = np.expand_dims(np.ascontiguousarray(img_np), axis=0)
            input_data = prepare_input_tensor(img_np, args.dtype, dev)

            start_time = time.time()
            output = vm["main"](input_data)
            end_time = time.time()
            inference_time = end_time - start_time

            logits = extract_logits(output, args.dtype, dev)
            if logits.shape[0] != 1:
                raise ValueError(f"Expected batch=1 from VM output, got shape {logits.shape}")

            num_model_classes = logits.shape[1]
            if label_mode is None:
                if num_model_classes == len(dataset.classes):
                    label_mode = "folder_index"
                elif mapped_labels is not None:
                    max_mapped_label = max(mapped_labels)
                    if max_mapped_label >= num_model_classes:
                        raise RuntimeError(
                            "Model output classes are too small for mapped ImageNet labels. "
                            f"Output classes={num_model_classes}, max mapped label={max_mapped_label}"
                        )
                    label_mode = "imagenet_1000_index"
                else:
                    raise RuntimeError(
                        "Cannot decide ground-truth label mapping automatically. "
                        "Provide --imagenet-class-index for ImageNet(1000) mapping."
                    )
                print(
                    f"[INFO] label_mode={label_mode}, "
                    f"num_model_classes={num_model_classes}, num_dataset_classes={len(dataset.classes)}"
                )

            if label_mode == "folder_index":
                true_label = int(local_label)
            else:
                true_label = int(mapped_labels[sample_idx])

            topk = min(5, logits.shape[1])
            top5_idx = np.argsort(logits[0])[-topk:][::-1]
            top1_idx = int(top5_idx[0])

            is_top1 = bool(top1_idx == true_label)
            is_top5 = bool(true_label in top5_idx)

            total += 1
            total_time += inference_time
            if is_top1:
                correct1 += 1
            if is_top5:
                correct5 += 1

            result = {
                "image_idx": image_idx,
                "image_path": image_path,
                "class_synset": dataset.classes[local_label],
                "dataset_local_label": int(local_label),
                "mapped_imagenet_label": int(mapped_labels[sample_idx]) if mapped_labels is not None else None,
                "label_mode": label_mode,
                "true_label": int(true_label),
                "top1_pred": top1_idx,
                "top5_preds": [int(x) for x in top5_idx],
                "correct_top1": is_top1,
                "correct_top5": is_top5,
                "inference_time_ms": float(inference_time * 1000.0),
            }
            save_single_result(results_dir, result, args.dtype, "None")

        except Exception as err:
            failed += 1
            print(f"[WARN] Failed on image_idx={image_idx}, path={image_path}: {err}")

    all_results = load_existing_results(results_dir, start_idx, end_idx, args.dtype)

    if all_results:
        total = len(all_results)
        correct1 = sum(1 for item in all_results if item["correct_top1"])
        correct5 = sum(1 for item in all_results if item["correct_top5"])
        total_time = sum(item["inference_time_ms"] for item in all_results) / 1000.0
        if label_mode is None:
            label_mode = all_results[0].get("label_mode", "unknown")

    dtype_suffix = normalize_dtype_suffix(args.dtype)
    summary_file = results_dir / f"summary_{dtype_suffix}_{start_idx:06d}_{end_idx:06d}.json"

    batch_summary = {
        "dtype": args.dtype,
        "model_path": str(model_path),
        "data_root": str(data_root),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "total_images": total,
        "skipped_images": skipped,
        "failed_images": failed,
        "correct_top1": correct1,
        "correct_top5": correct5,
        "top1_accuracy": 100.0 * correct1 / total if total > 0 else 0.0,
        "top5_accuracy": 100.0 * correct5 / total if total > 0 else 0.0,
        "total_time_seconds": total_time,
        "avg_inference_time_ms": total_time / total * 1000.0 if total > 0 else 0.0,
        "label_mode": label_mode,
        "num_dataset_classes": len(dataset.classes),
        "num_model_classes": num_model_classes,
        "imagenet_class_index": args.imagenet_class_index,
        "target": args.target,
    }

    with open(summary_file, "w") as f:
        json.dump(batch_summary, f, indent=2)

    top1 = batch_summary["top1_accuracy"]
    top5 = batch_summary["top5_accuracy"]
    avg_ms = batch_summary["avg_inference_time_ms"]

    print("\n" + "=" * 80)
    print("Batch Evaluation Results")
    print("=" * 80)
    print(f"Data Type: {args.dtype}")
    print(f"Image Range: {start_idx} to {end_idx}")
    print(f"Label Mode: {label_mode}")
    print(f"Total Images Processed: {total}")
    print(f"Skipped Images: {skipped}")
    print(f"Failed Images: {failed}")
    print(f"Top-1 Accuracy: {top1:.4f}%" if total > 0 else "Top-1 Accuracy: N/A")
    print(f"Top-5 Accuracy: {top5:.4f}%" if total > 0 else "Top-5 Accuracy: N/A")
    print(f"Average Inference Time: {avg_ms:.2f} ms/image" if total > 0 else "Average Inference Time: N/A")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Results saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
