import re
from collections.abc import Iterable

from tvm.runtime import DataType
from tvm.target.datatype import (
    get_type_registered,
    lower_call_pure_extern,
    lower_ite,
    register,
    register_min_func,
    register_op,
)
from tvm.tir.op import call_pure_extern

_POSIT_REGISTERED = False
_REGISTERED_ES = set()

# Keep existing code assignments for es=0..3.
_POSIT_TYPE_CODES = {
    0: 132,
    1: 133,
    2: 134,
    3: 135,
    4: 136,
    5: 137
}

_POSIT_DTYPE_RE = re.compile(r"custom\[posites(\d+)\](\d+)")


def _normalize_int_range(values, *, name, min_value=None, max_value=None):
    if values is None:
        return None
    if isinstance(values, int):
        values = [values]
    elif not isinstance(values, Iterable):
        raise TypeError(f"{name} must be int or iterable of ints, got {type(values)!r}")

    normalized = []
    for value in values:
        if not isinstance(value, int):
            raise TypeError(f"{name} items must be int, got {type(value)!r}")
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} items must be >= {min_value}, got {value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} items must be <= {max_value}, got {value}")
        normalized.append(value)

    if not normalized:
        raise ValueError(f"{name} cannot be empty")

    return sorted(set(normalized))


def _parse_es(type_name):
    match = re.fullmatch(r"posites(\d+)", type_name)
    if match is None:
        raise ValueError(f"unsupported posit type name: {type_name}")
    return int(match.group(1))


def _storage_dtype(dtype):
    t = DataType(dtype)
    if get_type_registered(t.type_code):
        storage = f"uint{t.bits}"
        if t.lanes > 1:
            storage += f"x{t.lanes}"
        return storage
    return dtype


def _posit_func(bits, es, suffix):
    return f"Posit{bits}es{es}{suffix}"


def _float_to_posit_name(src_bits, dst_bits, es):
    if src_bits in (16, 32):
        return f"FloatToPosit{dst_bits}es{es}"
    if src_bits == 64:
        return f"DoubleToPosit{dst_bits}es{es}"
    raise RuntimeError(f"unsupported float width {src_bits} for posit es={es}")


def _posit_to_float_name(src_bits, dst_bits, es):
    if dst_bits in (16, 32):
        return _posit_func(src_bits, es, "ToFloat")
    if dst_bits == 64:
        return _posit_func(src_bits, es, "ToDouble")
    raise RuntimeError(f"unsupported float width {dst_bits} for posit es={es}")


def _int_to_posit_name(_src_bits, dst_bits, es):
    return f"IntToPosit{dst_bits}es{es}"


def _posit_to_int_name(src_bits, _dst_bits, es):
    return _posit_func(src_bits, es, "ToInt")


def _uint_to_posit_name(src_bits, dst_bits, es):
    if src_bits == 1:
        return f"BoolToPosit{dst_bits}es{es}"
    if src_bits != dst_bits:
        raise RuntimeError(
            f"unsupported uint -> posit cast: uint{src_bits} -> posit{dst_bits}es{es} "
            f"(expected uint{dst_bits})"
        )
    return f"Uint{dst_bits}ToPosit{dst_bits}es{es}"


def _posit_to_uint_name(src_bits, dst_bits, es):
    if dst_bits == 1:
        return _posit_func(src_bits, es, "ToBool")
    if dst_bits != src_bits:
        raise RuntimeError(
            f"unsupported posit -> uint cast: posit{src_bits}es{es} -> uint{dst_bits} "
            f"(expected uint{src_bits})"
        )
    return _posit_func(src_bits, es, f"ToUint{src_bits}")


def _make_cast_lower(type_name, src_kind, dst_kind):
    es = _parse_es(type_name)

    def lower(op):
        src = DataType(op.value.dtype)
        dst = DataType(op.dtype)
        out_dtype = _storage_dtype(op.dtype)

        if src_kind == "posit" and dst_kind == "posit":
            if src.bits == dst.bits:
                return op.value
            # Avoid requiring O(n^2) PositXToPositY symbols.
            as_double = call_pure_extern("float64", _posit_to_float_name(src.bits, 64, es), op.value)
            return call_pure_extern(out_dtype, _float_to_posit_name(64, dst.bits, es), as_double)
        if src_kind == "float" and dst_kind == "posit":
            func_name = _float_to_posit_name(src.bits, dst.bits, es)
        elif src_kind == "posit" and dst_kind == "float":
            func_name = _posit_to_float_name(src.bits, dst.bits, es)
        elif src_kind == "int" and dst_kind == "posit":
            func_name = _int_to_posit_name(src.bits, dst.bits, es)
        elif src_kind == "posit" and dst_kind == "int":
            func_name = _posit_to_int_name(src.bits, dst.bits, es)
        elif src_kind == "uint" and dst_kind == "posit":
            func_name = _uint_to_posit_name(src.bits, dst.bits, es)
        elif src_kind == "posit" and dst_kind == "uint":
            func_name = _posit_to_uint_name(src.bits, dst.bits, es)
        else:
            raise RuntimeError(f"unsupported cast lowering path: {src_kind} -> {dst_kind}")

        return call_pure_extern(out_dtype, func_name, op.value)

    return lower


def _make_generic_lower(type_name, suffix):
    es = _parse_es(type_name)

    def lower(op):
        bits = DataType(op.dtype).bits
        out_dtype = _storage_dtype(op.dtype)
        func_name = _posit_func(bits, es, suffix)

        if hasattr(op, "args"):
            args = list(op.args)
        elif hasattr(op, "a") and hasattr(op, "b"):
            args = [op.a, op.b]
        else:
            args = [op.value]

        return call_pure_extern(out_dtype, func_name, *args)

    return lower


def _make_floatimm_lower(type_name):
    es = _parse_es(type_name)

    def lower(op):
        bits = DataType(op.dtype).bits
        return call_pure_extern(_storage_dtype(op.dtype), f"FloatToPosit{bits}es{es}", op.value)

    return lower


def _make_min_lower(type_name):
    es = _parse_es(type_name)

    def lower(num_bits):
        return call_pure_extern(
            f"custom[{type_name}]{num_bits}",
            _posit_func(num_bits, es, "MinValue"),
        )

    return lower


def _register_posit_family(type_name, type_code):
    register(type_name, type_code)

    register_op(_make_cast_lower(type_name, "posit", "posit"), "Cast", "llvm", type_name, type_name)
    register_op(_make_cast_lower(type_name, "float", "posit"), "Cast", "llvm", "float", type_name)
    register_op(_make_cast_lower(type_name, "posit", "float"), "Cast", "llvm", type_name, "float")
    register_op(_make_cast_lower(type_name, "uint", "posit"), "Cast", "llvm", "uint", type_name)
    register_op(_make_cast_lower(type_name, "posit", "uint"), "Cast", "llvm", type_name, "uint")
    register_op(_make_cast_lower(type_name, "int", "posit"), "Cast", "llvm", "int", type_name)
    register_op(_make_cast_lower(type_name, "posit", "int"), "Cast", "llvm", type_name, "int")

    register_op(_make_generic_lower(type_name, "Add"), "Add", "llvm", type_name)
    register_op(_make_generic_lower(type_name, "Sub"), "Sub", "llvm", type_name)
    register_op(_make_generic_lower(type_name, "Mul"), "Mul", "llvm", type_name)
    register_op(_make_generic_lower(type_name, "Div"), "Div", "llvm", type_name)
    register_op(_make_generic_lower(type_name, "Max"), "Max", "llvm", type_name)
    register_op(_make_generic_lower(type_name, "Min"), "Min", "llvm", type_name)
    register_op(_make_floatimm_lower(type_name), "FloatImm", "llvm", type_name)

    register_op(_make_generic_lower(type_name, "FMA"), "Call", "llvm", type_name, intrinsic_name="tir.fma")
    register_op(_make_generic_lower(type_name, "Sqrt"), "Call", "llvm", type_name, intrinsic_name="tir.sqrt")
    register_op(_make_generic_lower(type_name, "Pow"), "Call", "llvm", type_name, intrinsic_name="tir.pow")
    register_op(_make_generic_lower(type_name, "Exp"), "Call", "llvm", type_name, intrinsic_name="tir.exp")
    register_op(_make_generic_lower(type_name, "Log"), "Call", "llvm", type_name, intrinsic_name="tir.log")
    register_op(
        _make_generic_lower(type_name, "Sigmoid"),
        "Call",
        "llvm",
        type_name,
        intrinsic_name="tir.sigmoid",
    )
    register_op(_make_generic_lower(type_name, "Tanh"), "Call", "llvm", type_name, intrinsic_name="tir.tanh")
    register_op(_make_generic_lower(type_name, "Cos"), "Call", "llvm", type_name, intrinsic_name="tir.cos")
    register_op(_make_generic_lower(type_name, "Sin"), "Call", "llvm", type_name, intrinsic_name="tir.sin")
    register_op(_make_generic_lower(type_name, "Tan"), "Call", "llvm", type_name, intrinsic_name="tir.tan")
    register_op(_make_generic_lower(type_name, "Erf"), "Call", "llvm", type_name, intrinsic_name="tir.erf")
    register_op(
        _make_generic_lower(type_name, "Softmax"),
        "Call",
        "llvm",
        type_name,
        intrinsic_name="tir.softmax",
    )
    register_op(lower_ite, "Call", "llvm", type_name, intrinsic_name="tir.if_then_else")
    register_op(
        lower_call_pure_extern,
        "Call",
        "llvm",
        type_name,
        intrinsic_name="tir.call_pure_extern",
    )

    register_min_func(_make_min_lower(type_name), type_name)


def _posit_registered(bits=None, es=None):
    global _POSIT_REGISTERED

    bits = _normalize_int_range(bits, name="bits", min_value=3, max_value=64)
    if bits is None:
        bits = []

    es_list = _normalize_int_range(es, name="es", min_value=0, max_value=5)
    if es_list is None:
        es_list = sorted(_POSIT_TYPE_CODES)

    # Registration is per type code (i.e. per es family), not per bits value.
    for es_value in es_list:
        if es_value in _REGISTERED_ES:
            continue

        type_name = f"posites{es_value}"
        _register_posit_family(type_name, _POSIT_TYPE_CODES[es_value])
        _REGISTERED_ES.add(es_value)

    _POSIT_REGISTERED = bool(_REGISTERED_ES)


def ensure_posit_registered_for_dtype(dtype):
    """Lazy-register only the posit configuration referenced by dtype.

    Parameters
    ----------
    dtype : str
        TVM dtype string, e.g. "custom[posites1]10".
    """

    if dtype is None:
        return

    match = _POSIT_DTYPE_RE.fullmatch(str(dtype))
    if not match:
        return

    es_value = int(match.group(1))
    bits_value = int(match.group(2))
    _posit_registered(bits=[bits_value], es=[es_value])


__all__ = [
    "_posit_registered",
    "ensure_posit_registered_for_dtype",
]
