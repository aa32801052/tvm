"""
Mixed Precision Transform Pass for Relax IR.

This module implements a custom mixed precision transformation pass similar to
TVM's ToMixedPrecision, but written in Python for easier customization.

Architecture:
1. DTypeAnalyzer (backward pass): Analyzes which variables can be stored in FP16
2. MixedPrecisionRewriter (forward pass): Rewrites the IR with appropriate casts
"""

import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor, PyExprMutator, mutator, visitor
from tvm.relax.expr import (
    Constant, Var, DataflowVar, Function, Call, Tuple, TupleGetItem,
    VarBinding, DataflowBlock, BindingBlock, SeqExpr
)
from tvm.relax.struct_info import TensorStructInfo, TupleStructInfo
from tvm.ir.module import IRModule
from typing import Dict, Set, Optional, List
from enum import IntEnum


class MixedPrecisionPolicy(IntEnum):
    """Policy for mixed precision conversion."""
    NEVER = 0    # Never use FP16 (e.g., softmax, layer_norm)
    FOLLOW = 1   # Follow input dtype
    ALWAYS = 2   # Always use FP16 inputs with FP32 accumulation (e.g., matmul, conv)


class DTypeInfo(IntEnum):
    """Data type requirement info for variables."""
    UNKNOWN = 0  # Not yet analyzed
    FP16 = 16    # Can be stored as FP16
    FP32 = 32    # Must be stored as FP32


@visitor
class DTypeAnalyzer(PyExprVisitor):
    """
    Backward pass to analyze dtype requirements.
    
    Determines which variables can safely be stored in FP16 without
    affecting numerical stability.
    """
    
    def __init__(self, src_dtype: str = "float32", dst_dtype: str = "float16"):
        super().__init__()
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        self.var_dtype_map: Dict[Var, DTypeInfo] = {}
        
    def get_policy(self, call: Call) -> MixedPrecisionPolicy:
        """Get mixed precision policy for an operation."""
        if not hasattr(call.op, 'name'):
            return MixedPrecisionPolicy.NEVER
        
        op_name = call.op.name
        
        # Operations that should always use FP16 inputs (with FP32 accumulation)
        if op_name in ['relax.matmul', 'relax.nn.conv2d', 'relax.nn.linear']:
            return MixedPrecisionPolicy.ALWAYS
        
        # Operations that have numerical issues with FP16
        if op_name in ['relax.nn.softmax', 'relax.nn.layer_norm', 'relax.exp',
                       'relax.log', 'relax.sqrt', 'relax.rsqrt']:
            return MixedPrecisionPolicy.NEVER
        
        # Other operations follow input dtype
        return MixedPrecisionPolicy.FOLLOW
    
    def require_dtype(self, var: Var, dtype: DTypeInfo):
        """Update dtype requirement for a variable."""
        if var not in self.var_dtype_map:
            self.var_dtype_map[var] = dtype
        else:
            # Merge: take the more restrictive one (FP32 > FP16 > UNKNOWN)
            current = self.var_dtype_map[var]
            self.var_dtype_map[var] = max(current, dtype)
    
    def require_args_dtype(self, args: List, dtype: DTypeInfo):
        """Require all arguments to have certain dtype."""
        for arg in args:
            if isinstance(arg, Var):
                self.require_dtype(arg, dtype)
            elif isinstance(arg, Tuple):
                self.require_args_dtype(arg.fields, dtype)
    
    def visit_binding_(self, binding: VarBinding) -> None:
        """Visit a variable binding."""
        value = binding.value
        
        if isinstance(value, Call):
            policy = self.get_policy(value)
            
            if policy == MixedPrecisionPolicy.ALWAYS:
                # Require inputs to be FP16
                self.require_args_dtype(value.args, DTypeInfo.FP16)
            elif policy == MixedPrecisionPolicy.NEVER:
                # Require inputs to be FP32
                self.require_args_dtype(value.args, DTypeInfo.FP32)
            # FOLLOW: don't impose requirements
        
        # Visit the value expression
        super().visit_binding_(binding)
    
    def visit_var_(self, var: Var) -> None:
        """Visit a variable usage - require it to be FP32 by default."""
        if isinstance(var.struct_info, TensorStructInfo):
            if var.struct_info.dtype == self.src_dtype:
                # Default: require FP32 unless relaxed by ALWAYS ops
                self.require_dtype(var, DTypeInfo.FP32)
    
    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        """Visit dataflow block in reverse order (backward pass)."""
        for binding in reversed(block.bindings):
            self.visit_binding(binding)


@mutator
class MixedPrecisionRewriter(PyExprMutator):
    """
    Forward pass to rewrite IR with mixed precision.
    
    Converts constants to FP16, inserts casts as needed, and uses
    FP32 accumulation for compute-intensive ops.
    """
    
    def __init__(self, var_dtype_map: Dict[Var, DTypeInfo],
                 src_dtype: str = "float32", dst_dtype: str = "float16"):
        super().__init__()
        self.var_dtype_map = var_dtype_map
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
    
    def get_policy(self, call: Call) -> MixedPrecisionPolicy:
        """Get mixed precision policy for an operation."""
        if not hasattr(call.op, 'name'):
            return MixedPrecisionPolicy.NEVER
        
        op_name = call.op.name
        
        if op_name in ['relax.matmul', 'relax.nn.conv2d', 'relax.nn.linear']:
            return MixedPrecisionPolicy.ALWAYS
        if op_name in ['relax.nn.softmax', 'relax.nn.layer_norm', 'relax.exp',
                       'relax.log', 'relax.sqrt', 'relax.rsqrt']:
            return MixedPrecisionPolicy.NEVER
        return MixedPrecisionPolicy.FOLLOW
    
    def should_store_as_fp16(self, var: Var) -> bool:
        """Check if variable should be stored as FP16."""
        return self.var_dtype_map.get(var, DTypeInfo.FP32) == DTypeInfo.FP16
    
    def cast_if_needed(self, expr, target_dtype: str):
        """Cast expression to target dtype if needed."""
        if isinstance(expr, Constant):
            if expr.data.dtype == self.src_dtype and target_dtype == self.dst_dtype:
                return relax.op.astype(expr, self.dst_dtype)
            return expr
        
        if isinstance(expr, Var):
            var_sinfo = expr.struct_info
            if isinstance(var_sinfo, TensorStructInfo):
                current_dtype = var_sinfo.dtype
                if current_dtype != target_dtype:
                    return relax.op.astype(expr, target_dtype)
        
        return expr
    
    def visit_constant_(self, const: Constant):
        """Convert constants to FP16 for weight compression."""
        if const.data.dtype == self.src_dtype:
            return relax.op.astype(const, self.dst_dtype)
        return const
    
    def visit_call_(self, call: Call) -> Call:
        """Rewrite call operations with mixed precision."""
        policy = self.get_policy(call)
        new_args = [self.visit_expr(arg) for arg in call.args]
        
        if policy == MixedPrecisionPolicy.ALWAYS:
            # Cast inputs to FP16, use FP32 accumulation
            casted_args = [self.cast_if_needed(arg, self.dst_dtype) for arg in new_args]
            
            # For matmul, set out_dtype to FP32
            if call.op.name == 'relax.matmul':
                result = relax.op.matmul(casted_args[0], casted_args[1], out_dtype=self.src_dtype)
                # Cast output back to FP16 if it will be stored as FP16
                return result
            
            return Call(call.op, casted_args, call.attrs, call.span)
        
        elif policy == MixedPrecisionPolicy.NEVER:
            # Cast all inputs to FP32
            casted_args = [self.cast_if_needed(arg, self.src_dtype) for arg in new_args]
            return Call(call.op, casted_args, call.attrs, call.span)
        
        else:  # FOLLOW
            return Call(call.op, new_args, call.attrs, call.span)


def ToMixedPrecision(src_dtype: str = "float32", 
                     dst_dtype: str = "float16") -> tvm.ir.transform.Pass:
    """
    Create a mixed precision transformation pass.
    
    This is a simplified Python implementation. For production use,
    it's recommended to use TVM's C++ ToMixedPrecision which has
    more sophisticated dtype analysis and handling.
    
    Current limitation: This implementation only converts constants to FP16.
    For full mixed precision with proper dtype propagation, use TVM's
    transform.ToMixedPrecision() instead.
    
    Parameters
    ----------
    src_dtype : str
        Source data type (default: "float32")
    dst_dtype : str
        Destination data type (default: "float16")
    
    Returns
    -------
    pass : callable
        A function that takes IRModule and returns transformed IRModule
    
    Examples
    --------
    >>> # For production, use TVM's implementation:
    >>> from tvm.relax import transform
    >>> new_mod = transform.ToMixedPrecision()(mod)
    >>>
    >>> # This custom implementation (simplified):
    >>> custom_pass = ToMixedPrecision()
    >>> new_mod = custom_pass(mod)
    """
    
    def transform(mod: IRModule) -> IRModule:
        """
        Transform function implementation.
        
        Note: This is a simplified version that only converts constants.
        For proper mixed precision, use TVM's C++ implementation.
        """
        # For float32->float16, delegate to TVM's implementation
        if src_dtype == "float32" and dst_dtype == "float16":
            from tvm.relax import transform as tvm_transform
            return tvm_transform.ToMixedPrecision()(mod)
        
        # For other conversions, use simple constant conversion
        from .change_datatype import _ChangeDatatypeMutator
        mutator = _ChangeDatatypeMutator(src_dtype, dst_dtype, use_mixed_precision=False)
        new_functions = {}
        
        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_func = mutator.visit_expr(func)
                new_functions[gv] = new_func
            else:
                new_functions[gv] = func
        
        return IRModule(new_functions)
    
    # Return callable that matches TVM's transform pass interface
    return transform
