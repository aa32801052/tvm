import argparse
import os
import time
from pathlib import Path

import onnx
import tvm
from tvm.relax import transform
from tvm.relax.frontend.change_dataype import ChangeDatatype
from tvm.relax.frontend.onnx import from_onnx

from gpt2_dtype_utils import (
    parse_custom_dtype,
    register_custom_datatypes,
    validate_dtype_arg,
)

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# Helper function to retrieve all blocks in a TIR function
def get_all_blocks(func):
    blocks = []
    # Define a visitor to collect all blocks
    def visit_block(stmt):
        if isinstance(stmt, tvm.tirx.SBlock):
            blocks.append(stmt)
    tvm.tirx.stmt_functor.post_order_visit(func.body, visit_block)
    return blocks

# Function to apply automatic optimization to a given TIR function
def auto_optimize_func(sch, func, tile_sizes=(8, 16, 32, 16), use_vectorize=False):
    def custom_sort_key(item):
        index, value = item
        if isinstance(value, str):
            return (0, value)
        elif isinstance(value, int):
            return (1, -value)
        return (2, index)

    # Collect all block names in the function
    blocks = get_all_blocks(func)
    # Apply scheduling transformations to each block
    for block in blocks:
        block_name = block.name_hint
        # Get the target block
        blockRV = sch.get_sblock(block_name)
        # Get the loops of the block
        loops = sch.get_loops(blockRV)
        if len(loops) == 0:
            continue

        iter_vars = []
        for index, iter_var in enumerate(block.iter_vars):
            if iter_var.iter_type == iter_var.CommReduce:
                continue
            if isinstance(iter_var.dom.extent, tvm.tirx.IntImm):
                value = iter_var.dom.extent.value
            else:
                value = 'none'
            iter_vars.append(value)
        if len(iter_vars) > 0:
            iter_vars = [(index, value) for index, value in enumerate(iter_vars)]
            # print(iter_vars)
            sorted_list = sorted(iter_vars, key=custom_sort_key)
            # print("block:", block_name)
            # print("  #loops:", len(loops))
            # print("  iter_vars:", iter_vars)
            # print("  sorted_list:", sorted_list)
            # Check if there are enough loops to tile
            #if 'matmul' in block_name:

            reorder_list = []
            for i in range(len(sorted_list)):
                reorder_list.append(loops[sorted_list[i][0]])
            sch.reorder(*reorder_list)

            for i in range(0, len(sorted_list)):
                sch.parallel(loops[sorted_list[i][0]])
                sorted_list[i] = (-1, sorted_list[i][1])
                break

            # Use vectorize for float types
            if use_vectorize:
                for i in range(0, len(sorted_list)):
                    if (
                        sorted_list[i][0] != -1
                        and isinstance(sorted_list[i][1], int)
                        and sorted_list[i][1] > 1
                        and sorted_list[i][1] <= 256
                    ):
                        sch.vectorize(loops[sorted_list[i][0]])
                        sorted_list[i] = (-1, sorted_list[i][1])
                        break

            for i in range(0, len(sorted_list)):
                if (
                    sorted_list[i][0] != -1
                    and isinstance(sorted_list[i][1], int)
                    and sorted_list[i][1] <= 128
                ):
                    sch.unroll(loops[sorted_list[i][0]])
                    sorted_list[i] = (-1, sorted_list[i][1])
                    break


    #print(sch.mod.script())
    # Return the modified function from the schedule
    return sch.mod["main"]


# Function to optimize all TIR functions in the IR module
def optimize_ir_module(ir_module, use_vectorize=False):
    """ Optimize all TIR functions in the IR module.
    Args:
        ir_module: Input IR module
        use_vectorize: Whether to use vectorization
    """
    optimized_module = tvm.IRModule()

    # Iterate over each function in the IR module
    for name, func in ir_module.functions.items():
        if isinstance(func, tvm.tirx.PrimFunc):  # Only apply to TIR functions
            # Create a schedule and apply the optimizations
            sch = tvm.s_tir.Schedule(func)
            optimized_func = auto_optimize_func(sch, func, use_vectorize=use_vectorize)
            optimized_module[name] = optimized_func
        else:
            optimized_module[name] = func

    return optimized_module


def bind_symbolic_vars_if_present(mod, bindings):
    """Bind known GPT-2 symbolic variables while tolerating static-shape models."""

    for name, value in bindings.items():
        try:
            mod = tvm.relax.transform.BindSymbolicVars({name: value})(mod)
        except tvm.error.InternalError as err:
            if "did not correspond to any symbolic variables" not in str(err):
                raise
    return mod


def compile_model(
    onnx_path,
    dtype_converter=None,
    mixed_precision_config=None,
    use_vectorize=False,
    use_quire=False,
):
    """Load ONNX model and apply transformations

    Args:
        onnx_path: Path to ONNX model
        dtype_converter: Optional function to convert data types
        mixed_precision_config: Optional (source, destination, accumulator) dtype tuple
        use_vectorize: Whether to use vectorize optimization (for float types)
        use_quire: Whether to replace supported Posit matmuls with Quire extern calls
    """
    mod = onnx.load_model(onnx_path)
    mod = from_onnx(mod)
    mod = transform.DecomposeOpsForInference()(mod)
    mod = bind_symbolic_vars_if_present(mod, {"batch_size": 1, "sequence_length": 1})
    if dtype_converter:
        new_main = dtype_converter(mod)
        mod = tvm.IRModule({"main": new_main})
    if mixed_precision_config:
        mod = transform.ToMixedPrecisionCustom(*mixed_precision_config)(mod)
    mod = tvm.relax.transform.LegalizeOps()(mod)
    with tvm.transform.PassContext(
        opt_level=0, config={"tirx.disable_vectorize": not use_vectorize}
    ):
        # Fold constants first, then run TIR-level optimizations.
        mod = tvm.relax.transform.FoldConstant()(mod)
        mod = optimize_ir_module(mod, use_vectorize=use_vectorize)
        if use_quire:
            from tir_transform_matmul_to_quire import InjectQuireMatmulElem
            mod = InjectQuireMatmulElem()(mod)
    print(mod)
    return mod


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_dtype_converter(input_dtype, target_dtype):
    def _converter(mod):
        return ChangeDatatype(input_dtype, target_dtype)(mod)["main"]

    return _converter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile a GPT-2 ONNX model with a configurable datatype"
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./model/model.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="llvm",
        help="TVM target name or JSON configuration",
    )
    parser.add_argument(
        "--input-dtype",
        type=validate_dtype_arg,
        default="float32",
        help="Source dtype used by ChangeDatatype",
    )
    parser.add_argument(
        "--target-dtype",
        type=validate_dtype_arg,
        default="float32",
        help="Destination dtype. If equal to --input-dtype, dtype conversion is skipped",
    )
    parser.add_argument(
        "--use-vectorize",
        type=str2bool,
        default=False,
        help="Enable vectorization optimizations (true/false)",
    )
    parser.add_argument(
        "--use-mixed-precision",
        type=str2bool,
        default=False,
        help="Enable mixed precision for a registered custom datatype (true/false)",
    )
    parser.add_argument(
        "--mixed-precision-dtype",
        type=validate_dtype_arg,
        default=None,
        help="Lower-precision custom dtype used for mixed-precision computation",
    )
    parser.add_argument(
        "--mixed-precision-acc-dtype",
        type=validate_dtype_arg,
        default=None,
        help="Custom dtype used for mixed-precision accumulation",
    )
    parser.add_argument(
        "--use-quire",
        type=str2bool,
        default=False,
        help="Replace supported Posit matmuls with Quire extern calls (true/false)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model/GPT2_fp32.so",
        help="Output shared library path",
    )
    args = parser.parse_args()
    if args.use_mixed_precision:
        configured_dtypes = {
            "--target-dtype": args.target_dtype,
            "--mixed-precision-dtype": args.mixed_precision_dtype,
            "--mixed-precision-acc-dtype": args.mixed_precision_acc_dtype,
        }
        missing = [name for name, dtype in configured_dtypes.items() if dtype is None]
        if missing:
            parser.error("required with --use-mixed-precision: " + ", ".join(missing))

        parsed_dtypes = {
            name: parse_custom_dtype(dtype) for name, dtype in configured_dtypes.items()
        }
        non_custom = [name for name, parsed in parsed_dtypes.items() if parsed is None]
        if non_custom:
            parser.error(
                "custom mixed precision requires custom[name]bits for: "
                + ", ".join(non_custom)
            )
        families = {parsed[0] for parsed in parsed_dtypes.values()}
        if len(families) != 1:
            parser.error("all mixed-precision dtypes must use the same custom datatype name")
        source_bits = parsed_dtypes["--target-dtype"][1]
        compute_bits = parsed_dtypes["--mixed-precision-dtype"][1]
        accumulator_bits = parsed_dtypes["--mixed-precision-acc-dtype"][1]
        if compute_bits >= source_bits:
            parser.error(
                "--mixed-precision-dtype must use fewer bits than --target-dtype; "
                "--target-dtype is the full-precision custom source dtype"
            )
        if accumulator_bits < compute_bits:
            parser.error(
                "--mixed-precision-acc-dtype must use at least as many bits as "
                "--mixed-precision-dtype"
            )
        if args.use_quire:
            type_name = parsed_dtypes["--mixed-precision-dtype"][0]
            if not type_name.startswith("posites"):
                parser.error("Quire is only available for custom[posites<es>] datatypes")
            if (compute_bits, accumulator_bits) not in ((8, 32), (16, 32)):
                parser.error(
                    "mixed-precision Quire currently supports Posit8/16 inputs "
                    "with a Posit32 output"
                )
    return args


def main():
    args = parse_args()

    target = tvm.target.Target(args.target)

    registered_dtypes = [args.input_dtype, args.target_dtype]
    if args.use_mixed_precision:
        registered_dtypes.extend(
            (args.mixed_precision_dtype, args.mixed_precision_acc_dtype)
        )
    register_custom_datatypes(registered_dtypes, target=target.kind.name)

    dtype_converter = None
    if args.target_dtype != args.input_dtype:
        dtype_converter = build_dtype_converter(args.input_dtype, args.target_dtype)

    mixed_precision_config = None
    if args.use_mixed_precision:
        mixed_precision_config = (
            args.target_dtype,
            args.mixed_precision_dtype,
            args.mixed_precision_acc_dtype,
        )

    print("Compile configuration:")
    print(f"  onnx_path={args.onnx_path}")
    print(f"  target={args.target}")
    print(f"  input_dtype={args.input_dtype}")
    print(f"  target_dtype={args.target_dtype}")
    print(f"  use_mixed_precision={args.use_mixed_precision}")
    if mixed_precision_config:
        print(f"  mixed_precision_dtype={args.mixed_precision_dtype}")
        print(f"  mixed_precision_acc_dtype={args.mixed_precision_acc_dtype}")
    print(f"  use_vectorize={args.use_vectorize}")
    print(f"  use_quire={args.use_quire}")
    print(f"  output={args.output}")

    compile_start = time.time()
    mod = compile_model(
        args.onnx_path,
        dtype_converter=dtype_converter,
        mixed_precision_config=mixed_precision_config,
        use_vectorize=args.use_vectorize,
        use_quire=args.use_quire,
    )
    compile_time = time.time() - compile_start
    print(f"IR compile time: {compile_time:.2f} seconds")

    build_start = time.time()
    lib = tvm.relax.build(mod, target=target)
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f} seconds")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    export_start = time.time()
    lib.export_library(args.output)
    export_time = time.time() - export_start
    print(f"Export time: {export_time:.2f} seconds")
    print(f"Shared library written to: {args.output}")
    print("Done")


if __name__ == "__main__":
    main()
