"""
Custom Mixed Precision Transform with Custom DataType Support

This module extends TVM's ToMixedPrecision to support custom datatypes like posit.
"""

import tvm
from tvm import relax, DataType
from tvm.ir.module import IRModule
from typing import Optional, List, Union


def ToMixedPrecisionCustom(
    src_dtype: Union[str, DataType] = "float32",
    dst_dtype: Union[str, DataType] = "float16", 
    out_dtype: Optional[Union[str, DataType]] = None,
    fp16_input_names: Optional[List[str]] = None
) -> callable:
    """
    Extended mixed precision pass that supports custom datatypes.
    
    This is a wrapper that:
    1. For float32->float16: Uses TVM's optimized ToMixedPrecision
    2. For custom dtypes: Uses a custom implementation
    
    Parameters
    ----------
    src_dtype : str or DataType
        Source data type (e.g., "float32")
    dst_dtype : str or DataType
        Destination data type for weights (e.g., "float16", "custom[posites2]16")
    out_dtype : str or DataType, optional
        Output dtype for accumulation. If None, uses src_dtype.
    fp16_input_names : List[str], optional
        Names of function parameters to convert to dst_dtype
    
    Returns
    -------
    transform : callable
        A function that transforms IRModule
    
    Examples
    --------
    >>> # Standard FP32->FP16 (uses TVM's ToMixedPrecision)
    >>> transform = ToMixedPrecisionCustom("float32", "float16")
    >>> new_mod = transform(mod)
    
    >>> # Custom: FP32->Posit16 with FP32 accumulation
    >>> transform = ToMixedPrecisionCustom("float32", "custom[posites2]16", "float32")
    >>> new_mod = transform(mod)
    """
    
    # Normalize dtype strings to DataType objects
    if isinstance(src_dtype, str):
        src_dtype_str = src_dtype
    else:
        src_dtype_str = str(src_dtype)
    
    if isinstance(dst_dtype, str):
        dst_dtype_str = dst_dtype
    else:
        dst_dtype_str = str(dst_dtype)
    
    if out_dtype is None:
        out_dtype_str = src_dtype_str
    elif isinstance(out_dtype, str):
        out_dtype_str = out_dtype
    else:
        out_dtype_str = str(out_dtype)
    
    # Validate dtype combination
    from tvm.relax.frontend.change_datatype import _validate_mixed_precision_dtypes
    is_valid, error_msg = _validate_mixed_precision_dtypes(
        src_dtype_str, dst_dtype_str, out_dtype_str
    )
    if not is_valid:
        raise ValueError(
            f"Invalid dtype combination for mixed precision:\n{error_msg}\n\n"
            f"For simple dtype conversion (without mixed precision restrictions), "
            f"use ChangeDatatype('{src_dtype_str}', '{dst_dtype_str}', use_mixed_precision=False)"
        )
    
    # Check if we can use TVM's optimized implementation
    is_fp32_to_fp16 = (
        src_dtype_str == "float32" and 
        dst_dtype_str == "float16" and
        out_dtype_str == "float32"
    )
    
    if is_fp32_to_fp16:
        # Use TVM's optimized C++ implementation
        from tvm.relax import transform
        return transform.ToMixedPrecision(out_dtype_str, fp16_input_names)
    else:
        # Use custom implementation for non-standard dtypes
        return _CustomMixedPrecisionTransform(
            src_dtype_str, dst_dtype_str, out_dtype_str, fp16_input_names
        )


class _CustomMixedPrecisionTransform:
    """
    Custom mixed precision transform for non-standard datatypes.
    
    Strategy:
    1. Convert all constants (weights) to dst_dtype
    2. Keep computation in src_dtype for numerical stability
    3. Use out_dtype for accumulation in matmul/conv
    """
    
    def __init__(self, src_dtype: str, dst_dtype: str, out_dtype: str,
                 fp16_input_names: Optional[List[str]] = None):
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        self.out_dtype = out_dtype
        self.fp16_input_names = fp16_input_names or []
    
    def __call__(self, mod: IRModule) -> IRModule:
        """Apply custom mixed precision transform."""
        from tvm.relax.expr_functor import PyExprMutator, mutator
        from tvm.relax.expr import Constant, Call, Function, Var
        from tvm.relax.struct_info import TensorStructInfo
        
        @mutator
        class CustomMixedPrecisionMutator(PyExprMutator):
            def __init__(self, src_dtype, dst_dtype, out_dtype, fp16_input_names):
                super().__init__()
                self.src_dtype = src_dtype
                self.dst_dtype = dst_dtype
                self.out_dtype = out_dtype
                self.fp16_input_names_set = set(fp16_input_names)
            
            def visit_constant_(self, const: Constant):
                """Convert constants (weights) to dst_dtype."""
                if const.data.dtype == self.src_dtype:
                    return relax.op.astype(const, self.dst_dtype)
                return const
            
            def visit_call_(self, call: Call):
                """Handle operations with custom dtype support."""
                new_args = [self.visit_expr(arg) for arg in call.args]
                
                # Check if this is a matmul operation
                if hasattr(call.op, 'name') and call.op.name == 'relax.matmul':
                    # Cast arguments to dst_dtype if they're constants
                    # Use out_dtype for accumulation
                    casted_args = []
                    for arg in new_args:
                        if isinstance(arg, Constant):
                            # Already converted in visit_constant_
                            casted_args.append(arg)
                        elif isinstance(arg, Var):
                            # Keep variables as-is (computation stays in src_dtype)
                            # They will be cast automatically by the op
                            casted_args.append(arg)
                        else:
                            casted_args.append(arg)
                    
                    # Set out_dtype for accumulation
                    try:
                        return relax.op.matmul(
                            casted_args[0], casted_args[1], 
                            out_dtype=self.out_dtype
                        )
                    except:
                        # Fallback if out_dtype not supported
                        return Call(call.op, casted_args, call.attrs, call.span)
                
                # For other operations, use default behavior
                return Call(call.op, new_args, call.attrs, call.span)
            
            def visit_function_(self, fn: Function):
                """Transform function, optionally converting parameter types."""
                # Transform parameters if specified
                new_params = []
                for param in fn.params:
                    if param.name_hint in self.fp16_input_names_set:
                        # Convert this parameter to dst_dtype
                        if isinstance(param.struct_info, TensorStructInfo):
                            if param.struct_info.dtype == self.src_dtype:
                                new_sinfo = TensorStructInfo(
                                    param.struct_info.shape, self.dst_dtype
                                )
                                new_param = Var(param.name_hint, new_sinfo)
                                self.set_var_remap(param.vid, new_param)
                                new_params.append(new_param)
                                continue
                    new_params.append(param)
                
                # Transform body
                new_body = self.visit_expr(fn.body)
                
                # Return new function
                return Function(
                    new_params, new_body, 
                    ret_struct_info=fn.ret_struct_info, 
                    attrs=fn.attrs
                )
        
        # Apply transformation
        mutator = CustomMixedPrecisionMutator(
            self.src_dtype, self.dst_dtype, self.out_dtype, self.fp16_input_names
        )
        
        new_functions = {}
        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_func = mutator.visit_expr(func)
                new_functions[gv] = new_func
            else:
                new_functions[gv] = func
        
        return IRModule(new_functions)


# Convenience function
def to_mixed_precision_posit(
    posit_bits: int = 16,
    posit_es: int = 2,
    src_dtype: str = "float32",
    out_dtype: str = "float32"
) -> callable:
    """
    Create mixed precision transform for Posit datatype.
    
    Note: This function requires src_dtype and out_dtype to also be posit types
    with the same es parameter, as mixed precision cannot mix float and posit.
    
    Parameters
    ----------
    posit_bits : int
        Bit width of posit (e.g., 16, 32)
    posit_es : int
        Exponent size (es) parameter
    src_dtype : str
        Source dtype. Should be a posit type with the same es (e.g., "custom[posites2]32")
    out_dtype : str
        Accumulation dtype. Should be a posit type with the same es (e.g., "custom[posites2]32")
    
    Returns
    -------
    transform : callable
        Transform function
    
    Examples
    --------
    >>> # Posit32 model with Posit16 weights, Posit32 accumulation
    >>> transform = to_mixed_precision_posit(16, 2, "custom[posites2]32", "custom[posites2]32")
    >>> new_mod = transform(mod)
    
    Raises
    ------
    ValueError
        If src_dtype or out_dtype are not posit types with the same es parameter
    """
    dst_dtype = f"custom[posites{posit_es}]{posit_bits}"
    return ToMixedPrecisionCustom(src_dtype, dst_dtype, out_dtype)
