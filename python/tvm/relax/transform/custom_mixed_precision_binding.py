"""
Python bindings for custom datatype mixed precision pass.

This module provides Python access to the C++ ToMixedPrecisionCustom pass
that supports custom datatypes like Posit.
"""

from typing import Optional, List
from tvm import ir
from tvm.ffi import get_global_func


def ToMixedPrecisionCustom(
    src_dtype: str,
    dst_dtype: str,
    acc_dtype: str,
    dst_input_names: Optional[List[str]] = None
) -> ir.transform.Pass:
    """
    Create a mixed precision pass with custom datatype support.
    
    This pass enables mixed precision transformation for custom datatypes,
    replacing the hardcoded float32/float16 logic with configurable types.
    
    Parameters
    ----------
    src_dtype : str
        Source datatype (e.g., "float32", "custom[posites2]32")
    dst_dtype : str
        Destination datatype for computation (e.g., "float16", "custom[posites2]16")
    acc_dtype : str
        Accumulator datatype for operations like matmul (e.g., "float32", "custom[posites2]32")
    dst_input_names : Optional[List[str]]
        Optional list of parameter names to convert to dst_dtype
        
    Returns
    -------
    pass : ir.transform.Pass
        The mixed precision transformation pass
        
    Examples
    --------
    >>> # Posit32 -> Posit16 mixed precision with Posit32 accumulation
    >>> pass1 = ToMixedPrecisionCustom(
    ...     "custom[posites2]32",
    ...     "custom[posites2]16", 
    ...     "custom[posites2]32"
    ... )
    >>> 
    >>> # Float32 -> Float16 (equivalent to standard ToMixedPrecision)
    >>> pass2 = ToMixedPrecisionCustom("float32", "float16", "float32")
    """
    func = get_global_func("relax.transform.ToMixedPrecisionCustom")
    return func(src_dtype, dst_dtype, acc_dtype, dst_input_names)


def ToMixedPrecisionPosit32ToPosit16(
    es: int = 2,
    dst_input_names: Optional[List[str]] = None
) -> ir.transform.Pass:
    """
    Convenience function for Posit32 -> Posit16 mixed precision.
    
    Parameters
    ----------
    es : int
        Exponent size for Posit (default: 2)
    dst_input_names : Optional[List[str]]
        Optional list of parameter names to convert to Posit16
        
    Returns
    -------
    pass : ir.transform.Pass
        The mixed precision transformation pass
        
    Examples
    --------
    >>> # Standard Posit32 -> Posit16 with es=2
    >>> pass1 = ToMixedPrecisionPosit32ToPosit16()
    >>> 
    >>> # With specific input parameters
    >>> pass2 = ToMixedPrecisionPosit32ToPosit16(
    ...     es=2,
    ...     dst_input_names=["weight", "bias"]
    ... )
    """
    func = get_global_func("relax.transform.ToMixedPrecisionPosit32ToPosit16")
    return func(es, dst_input_names)


def ToMixedPrecisionPosit16ToPosit8(
    es: int = 2,
    dst_input_names: Optional[List[str]] = None
) -> ir.transform.Pass:
    """
    Convenience function for Posit16 -> Posit8 mixed precision.
    
    Parameters
    ----------
    es : int
        Exponent size for Posit (default: 2)
    dst_input_names : Optional[List[str]]
        Optional list of parameter names to convert to Posit8
        
    Returns
    -------
    pass : ir.transform.Pass
        The mixed precision transformation pass
        
    Examples
    --------
    >>> # Standard Posit16 -> Posit8 with es=2
    >>> pass1 = ToMixedPrecisionPosit16ToPosit8()
    """
    func = get_global_func("relax.transform.ToMixedPrecisionPosit16ToPosit8")
    return func(es, dst_input_names)


def ToMixedPrecisionPosit(
    es: int,
    src_bits: int,
    dst_bits: int,
    acc_bits: int,
    dst_input_names: Optional[List[str]] = None,
) -> ir.transform.Pass:
    """Generic Posit mixed precision pass for arbitrary configurations.

    Parameters
    ----------
    es : int
        Exponent size for posit.
    src_bits : int
        Source posit bit width.
    dst_bits : int
        Destination posit bit width.
    acc_bits : int
        Accumulator posit bit width.
    dst_input_names : Optional[List[str]]
        Optional list of parameter names to convert to destination dtype.
    """

    func = get_global_func("relax.transform.ToMixedPrecisionPosit")
    return func(es, src_bits, dst_bits, acc_bits, dst_input_names)
