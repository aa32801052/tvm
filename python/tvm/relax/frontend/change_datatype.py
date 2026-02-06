"""
Data type conversion utilities for Relax IR.

This module provides functionality to convert tensor data types in Relax IR modules.
Supports conversion between standard types (float32, float16, etc.) and custom data types
(e.g., custom[myfloat]32, custom[posites2]16, etc.).
"""

from operator import call
import tvm
from tvm import relax
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.expr import Constant, Var, DataflowVar, Function, VarBinding, DataflowBlock, Call
from tvm.relax.struct_info import TensorStructInfo, TupleStructInfo, StructInfo
from tvm import tir
from tvm.relax.expr import Tuple

@mutator
class ChangeDatatype(PyExprMutator):
    """
    Data type conversion mutator for Relax IR.
    
    This class converts tensors from source dtype to destination dtype in a Relax module.
    It supports both standard types (float32, float16, int32, etc.) and custom data types
    (e.g., custom[myfloat]32, custom[posites2]16, etc.).
    
    Parameters
    ----------
    src : str
        Source data type (e.g., "float32", "custom[myfloat]32")
    dst : str
        Destination data type (e.g., "float16", "custom[myfloat]16")
    mod : IRModule, optional
        The module to convert (not used in current implementation)
    
    Examples
    --------
    >>> # Convert from float32 to custom myfloat32
    >>> converter = ChangeDatatype("float32", "custom[myfloat]32")
    >>> new_func = converter.visit_expr(mod["main"])
    """
    def __init__(self, src: str, dst: str, mod=None) -> None:
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

    def visit_constant_(self, const: Constant) -> relax.Expr:
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
        """
        Convert call expressions, handling special operations.
        
        Special handling for:
        - relax.sqrt: Wraps output with astype
        - relax.astype: Converts dtype parameter
        - Operations with out_dtype attribute
        - relax.nn.conv2d and relax.matmul: Reconstructs with new attrs
        """
        new_args = [self.visit_expr(arg) for arg in call.args]
        
        # Extract attributes to dictionary
        attrs = {}
        if call.attrs is not None:
            attrs = {
                k: getattr(call.attrs, k)
                for k in dir(call.attrs)
                if not k.startswith("_") and not callable(getattr(call.attrs, k))
            }
            
        # Special handling for sqrt: wrap output with astype
        if call.op.name == "relax.sqrt":
            out = relax.Call(call.op, new_args, call.attrs)
            return relax.op.astype(out, self.dst)

        # Special handling for astype: convert dtype parameter
        if call.op.name == "relax.astype":
            orig = attrs.get("dtype")
            new_dtype = self.dst if orig == self.src else orig
            return relax.op.astype(new_args[0], new_dtype)

        # Update out_dtype attribute if it matches source dtype
        if attrs.get("out_dtype") == self.src:
            attrs["out_dtype"] = self.dst

        # Reconstruct specific operations with updated attributes
        if call.op.name == "relax.nn.conv2d":
            return relax.op.nn.conv2d(*new_args, **attrs)

        if call.op.name == "relax.matmul":
            return relax.op.matmul(*new_args, **attrs)
        
        # Default: return call with original attrs
        return Call(call.op, new_args, call.attrs, call.span)

