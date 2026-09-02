"""Compile an ONNX CNN with an optional custom datatype."""

import argparse
import os
import subprocess
from pathlib import Path

import onnx
import tvm
from tvm.relax.frontend.change_dataype import ChangeDatatype
from tvm.relax.frontend.onnx import from_onnx

from custom_dtype_runtime import (
    configure_custom_datatypes,
    is_custom_dtype,
    register_custom_datatypes,
    validate_dtype_arg,
)

TARGET = "llvm"
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model"


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y", "on"):
        return True
    if value.lower() in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile an ONNX CNN with TVM and an optional custom datatype"
    )
    parser.add_argument(
        "--onnx-model-path",
        default=str(MODEL_DIR / "imagenet100_mobilenet.onnx"),
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--input-dtype",
        type=validate_dtype_arg,
        default="float32",
        help="Source model dtype, for example float32 or custom[myfloat]32",
    )
    parser.add_argument(
        "--target-dtype",
        type=validate_dtype_arg,
        default=None,
        help="Destination dtype; defaults to --input-dtype",
    )
    parser.add_argument(
        "--use-optimization",
        choices=("none", "database", "tune"),
        default="none",
        help="Meta-schedule optimization mode",
    )
    parser.add_argument(
        "--work-dir", default="./tuning_logs", help="Meta-schedule work directory"
    )
    parser.add_argument(
        "--max-trials-global",
        type=int,
        default=3000,
        help="Maximum number of tuning trials",
    )
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Output directory")
    parser.add_argument(
        "--model-name", default="imagenet100_mobilenet.so", help="Output filename"
    )
    parser.add_argument("--target", default=TARGET, help="TVM target string")
    parser.add_argument(
        "--dump-tir", type=str2bool, default=False, help="Dump lowered TIR (true/false)"
    )
    parser.add_argument(
        "--strip", type=str2bool, default=True, help="Strip the exported library (true/false)"
    )
    return parser.parse_args()


def convert_model_dtype(mod, input_dtype, target_dtype):
    """Convert the imported float32 ONNX module through the requested dtypes."""

    if input_dtype == "float32" and target_dtype == "float32":
        return mod

    if input_dtype != "float32":
        mod = ChangeDatatype("float32", input_dtype)(mod)
    if target_dtype != input_dtype:
        mod = ChangeDatatype(input_dtype, target_dtype)(mod)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    return tvm.relax.transform.UpdateParamType(lambda var: var.ty)(mod)


def prepare_for_meta_schedule(mod, target):
    pipeline = tvm.transform.Sequential(
        [
            tvm.relax.transform.LegalizeOps(),
            tvm.relax.transform.AnnotateTIROpPattern(),
            tvm.relax.transform.FoldConstant(),
            tvm.relax.transform.FuseOps(),
            tvm.relax.transform.FuseTIR(),
        ]
    )
    with tvm.target.Target(target):
        return pipeline(mod)


def debug_dump_tir(mod, target):
    preview_mod = prepare_for_meta_schedule(mod, target)
    tir_funcs = {
        gv: func
        for gv, func in preview_mod.functions.items()
        if isinstance(func, tvm.tirx.PrimFunc)
    }
    if not tir_funcs:
        print("[Compile Debug] No TIR PrimFunc found.")
        return

    target = tvm.target.Target(target)
    tir_mod = tvm.IRModule(tir_funcs)
    tir_mod = tvm.tirx.transform.BindTarget(target)(tir_mod)
    tir_mod = tvm.tirx.transform.LowerCustomDatatypes()(tir_mod)
    print("[Compile Debug] Lowered TIR PrimFuncs:")
    for global_var, func in tir_mod.functions.items():
        if isinstance(func, tvm.tirx.PrimFunc):
            print(f"\n# --- {global_var.name_hint} ---")
            print(func.script())


def compile_model(mod, target, optimization, work_dir, max_trials_global, dump_tir):
    """Build directly or use a meta-schedule database."""

    with tvm.transform.PassContext(config={"tirx.disable_vectorize": True}):
        if optimization in ("database", "tune"):
            mod = prepare_for_meta_schedule(mod, target)

        if optimization == "tune":
            from tvm.s_tir.meta_schedule.builder import LocalBuilder
            from tvm.s_tir.meta_schedule.relax_integration import tune_relax
            from tvm.s_tir.meta_schedule.runner import LocalRunner

            tune_relax(
                mod=mod,
                params={},
                target=target,
                work_dir=work_dir,
                max_trials_global=max_trials_global,
                builder=LocalBuilder(max_workers=8, initializer=register_custom_datatypes),
                runner=LocalRunner(initializer=register_custom_datatypes),
            )

        if optimization in ("database", "tune"):
            print(f"Applying tuning database from {work_dir}")
            with tvm.target.Target(target):
                mod = tvm.relax.transform.MetaScheduleApplyDatabase(work_dir)(mod)

        executable = tvm.relax.build(mod, target=target)

    if dump_tir:
        debug_dump_tir(mod, target)
    return executable


def save_model(executable, output_path, strip):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    executable.export_library(str(output_path))
    if strip:
        subprocess.run(
            [os.environ.get("STRIP", "strip"), "--strip-unneeded", str(output_path)],
            check=True,
        )
    print(f"Saved compiled model to: {output_path}")


def infer_shape_dict(onnx_model):
    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    shape_dict = {}
    for value_info in onnx_model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = [dim.dim_value for dim in tensor_type.shape.dim]
        if dims and all(dims):
            shape_dict[value_info.name] = dims
    return shape_dict


def main():
    args = parse_args()
    input_dtype = args.input_dtype
    target_dtype = args.target_dtype or input_dtype

    model_path = Path(args.onnx_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    custom_dtypes = [dtype for dtype in (input_dtype, target_dtype) if is_custom_dtype(dtype)]
    if custom_dtypes:
        target_kind = tvm.target.Target(args.target).kind.name
        configure_custom_datatypes(custom_dtypes, target=target_kind)

    print(f"Loading ONNX model from: {model_path}")
    onnx_model = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))
    shape_dict = infer_shape_dict(onnx_model)
    print(f"Using ONNX shape_dict={shape_dict}")
    mod = from_onnx(onnx_model, shape_dict=shape_dict or None)
    mod = convert_model_dtype(mod, input_dtype, target_dtype)

    print("Compile configuration:")
    print(f"  input_dtype={input_dtype}")
    print(f"  target_dtype={target_dtype}")
    print(f"  optimization={args.use_optimization}")

    executable = compile_model(
        mod,
        target=args.target,
        optimization=args.use_optimization,
        work_dir=args.work_dir,
        max_trials_global=args.max_trials_global,
        dump_tir=args.dump_tir,
    )
    save_model(executable, Path(args.model_dir) / args.model_name, args.strip)


if __name__ == "__main__":
    main()
