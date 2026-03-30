"""Data type conversion utilities for Relax IR.

This module supports both:
1) Simple dtype conversion for all tensor values.
2) Mixed precision conversion where compute and accumulation dtypes are controlled.
"""

import re

from tvm import relax
from tvm.ir.module import IRModule
from tvm.relax.expr import Call, Constant, DataflowVar, Function, Tuple, Var
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.struct_info import StructInfo, TensorStructInfo, TupleStructInfo

try:
    from tvm.relax.transform.custom_mixed_precision_binding import ToMixedPrecisionCustom

    HAS_CUSTOM_MP = True
except ImportError:
    HAS_CUSTOM_MP = False


def _parse_dtype_info(dtype_str: str) -> dict:
    """Parse dtype string and return family/parameter metadata."""
    if dtype_str.startswith("custom["):
        match = re.match(r"custom\[posites(\d+)\](\d+)$", dtype_str)
        if match:
            return {
                "family": "posit",
                "bits": int(match.group(2)),
                "es": int(match.group(1)),
                "original": dtype_str,
            }
        return {"family": "other", "bits": None, "es": None, "original": dtype_str}

    match = re.match(r"float(\d+)$", dtype_str)
    if match:
        return {
            "family": "float",
            "bits": int(match.group(1)),
            "es": None,
            "original": dtype_str,
        }

    return {"family": "other", "bits": None, "es": None, "original": dtype_str}


def _validate_mixed_precision_dtypes(src: str, dst: str, out_dtype: str) -> tuple:
    """Validate dtype compatibility for mixed precision mode."""
    src_info = _parse_dtype_info(src)
    dst_info = _parse_dtype_info(dst)
    out_info = _parse_dtype_info(out_dtype)

    if not (
        src_info["family"] == "posit"
        and dst_info["family"] == "posit"
        and out_info["family"] == "posit"
    ):
        return (
            False,
            (
                "Mixed precision only supports posit dtypes with the same configuration. "
                f"Got: src={src}, dst={dst}, out_dtype={out_dtype}."
            ),
        )

    posit_es_values = []
    if src_info["family"] == "posit":
        posit_es_values.append(src_info["es"])
    if dst_info["family"] == "posit":
        posit_es_values.append(dst_info["es"])
    if out_info["family"] == "posit":
        posit_es_values.append(out_info["es"])
    if posit_es_values and len(set(posit_es_values)) > 1:
            return (
                False,
                (
                    "Mixed precision with posit dtypes requires the same es value. "
                    f"Got: src={src}, dst={dst}, out_dtype={out_dtype}."
                ),
            )

    if src_info["bits"] is not None and dst_info["bits"] is not None:
        if dst_info["bits"] > src_info["bits"]:
            return (
                False,
                (
                    "Mixed precision expects destination precision not greater than source. "
                    f"Got src={src} -> dst={dst}."
                ),
            )

    if out_info["bits"] is not None and dst_info["bits"] is not None:
        if out_info["bits"] < dst_info["bits"]:
            return (
                False,
                (
                    "Accumulation dtype should have precision >= destination dtype. "
                    f"Got dst={dst}, out_dtype={out_dtype}."
                ),
            )

    return True, None


@mutator
class _ChangeDatatypeMutator(PyExprMutator):
    """Internal mutator for simple dtype conversion."""

    def __init__(self, src: str, dst: str) -> None:
        super().__init__()
        self.src = src
        self.dst = dst

    def _convert_struct_info(self, struct_info: StructInfo) -> StructInfo:
        """
        Convert struct info dtype from source to destination.
        
        Recursively converts TensorStructInfo and TupleStructInfo.
        """
        if isinstance(struct_info, TensorStructInfo):
            if struct_info.dtype == self.src:
                return TensorStructInfo(struct_info.shape, self.dst)
            return struct_info
        elif isinstance(struct_info, TupleStructInfo):
            new_fields = [self._convert_struct_info(field) for field in struct_info.fields]
            return TupleStructInfo(new_fields)
        else:
            return struct_info

    def visit_constant_(self, const: Constant):
        """Convert constant tensors to destination dtype."""
        if const.data.dtype == self.src:
            return relax.op.astype(const, self.dst)
        return const

    def visit_var_def_(self, var: Var) -> Var:
        """Convert variable definitions to destination dtype."""
        si = var.struct_info
        if isinstance(si, TensorStructInfo) and si.dtype == self.src:
            new_si = TensorStructInfo(si.shape, self.dst)
            new_var = Var(var.name_hint, new_si)
            self.set_var_remap(var.vid, new_var)
            return new_var
        return var

    def visit_dataflow_var_def_(self, var: DataflowVar) -> DataflowVar:
        """Convert dataflow variable definitions to destination dtype."""
        si = var.struct_info
        if isinstance(si, TensorStructInfo) and si.dtype == self.src:
            new_si = TensorStructInfo(si.shape, self.dst)
            new_var = DataflowVar(var.name_hint, new_si)
            self.set_var_remap(var.vid, new_var)
            return new_var
        return var

    def visit_var_(self, var: Var) -> Var:
        """Visit variable references, returning remapped variables."""
        remapped = self.get_var_remap(var.vid)
        return remapped if remapped is not None else var

    def visit_tuple_(self, tuple_expr: Tuple) -> Tuple:
        """Visit tuple expressions, converting all fields."""
        new_fields = [self.visit_expr(field) for field in tuple_expr.fields]
        return Tuple(new_fields)

    def visit_function_(self, fn: Function) -> Function:
        """Convert function parameters, body, and return type."""
        new_params = [self.visit_var_def(p) for p in fn.params]
        new_body = self.visit_expr(fn.body)
        new_ret = self._convert_struct_info(fn.ret_struct_info)
        return Function(new_params, new_body, ret_struct_info=new_ret, attrs=fn.attrs)

    def visit_call_(self, call: Call) -> Call:
        """Convert call expressions and preserve known dtype attrs."""
        new_args = [self.visit_expr(arg) for arg in call.args]

        attrs = {}
        if call.attrs is not None:
            attrs = {
                k: getattr(call.attrs, k)
                for k in dir(call.attrs)
                if not k.startswith("_") and not callable(getattr(call.attrs, k))
            }

        if call.op.name == "relax.astype":
            orig = attrs.get("dtype")
            new_dtype = self.dst if orig == self.src else orig
            return relax.op.astype(new_args[0], new_dtype)

        if attrs.get("out_dtype") == self.src:
            attrs["out_dtype"] = self.dst

        if call.op.name == "relax.nn.conv2d":
            return relax.op.nn.conv2d(*new_args, **attrs)

        if call.op.name == "relax.matmul":
            return relax.op.matmul(*new_args, **attrs)

        return Call(call.op, new_args, call.attrs, call.span)


class ChangeDatatype:
    """Convert Relax IR dtype with optional mixed precision support.

    Parameters
    ----------
    src : str
        Source dtype string.
    dst : str
        Destination dtype string.
    mod : IRModule, optional
        Kept for backward compatibility and ignored by this implementation.
    use_mixed_precision : bool
        Enable mixed precision pass.
    out_dtype : str, optional
        Accumulation dtype used by mixed precision pass. Defaults to src.
    """

    def __init__(
        self,
        src: str,
        dst: str,
        mod=None,
        use_mixed_precision: bool = False,
        out_dtype: str = None,
    ) -> None:
        self.src = src
        self.dst = dst
        self.use_mixed_precision = use_mixed_precision
        self.out_dtype = out_dtype or src
        self._simple_mutator = _ChangeDatatypeMutator(src, dst)

    def visit_expr(self, expr):
        """Backward-compatible API for scripts that call visit_expr directly."""
        if self.use_mixed_precision:
            raise ValueError("visit_expr is only available in simple conversion mode.")
        return self._simple_mutator.visit_expr(expr)

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply dtype conversion to the full IRModule."""
        if self.use_mixed_precision:
            is_valid, error_msg = _validate_mixed_precision_dtypes(
                self.src, self.dst, self.out_dtype
            )
            if not is_valid:
                raise ValueError(
                    f"Invalid dtype combination for mixed precision:\n{error_msg}\n\n"
                    f"For simple dtype conversion, use "
                    f"ChangeDatatype('{self.src}', '{self.dst}', use_mixed_precision=False)."
                )

            if HAS_CUSTOM_MP:
                return ToMixedPrecisionCustom(self.src, self.dst, self.out_dtype)(mod)

            from tvm.relax.transform.custom_mixed_precision import (
                ToMixedPrecisionCustom as PyToMixedPrecisionCustom,
            )

            return PyToMixedPrecisionCustom(self.src, self.dst, self.out_dtype)(mod)

        new_functions = {}
        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_functions[gv] = self._simple_mutator.visit_expr(func)
            else:
                new_functions[gv] = func
        return IRModule(new_functions)

