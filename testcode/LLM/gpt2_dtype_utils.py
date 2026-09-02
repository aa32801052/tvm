import argparse
import ctypes
import hashlib
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import tvm


_CUSTOM_DTYPE_RE = re.compile(r"^custom\[([A-Za-z_][A-Za-z0-9_]*)\](\d+)$")
_wrapper_library = None
_converter_cache = {}
_registered_extra_casts = set()


def parse_custom_dtype(dtype_str):
    if not isinstance(dtype_str, str):
        return None
    match = _CUSTOM_DTYPE_RE.fullmatch(dtype_str)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def custom_type_code(type_name):
    """Derive a stable DLPack custom type code from a datatype name."""

    digest = hashlib.blake2s(type_name.encode("utf-8"), digest_size=2).digest()
    return 129 + int.from_bytes(digest, byteorder="little") % 127


def _load_wrapper_library():
    """Load custom datatype wrapper symbols into the process-global namespace."""

    global _wrapper_library
    if _wrapper_library is not None:
        return _wrapper_library

    directories = []
    if library_path := os.environ.get("TVM_LIBRARY_PATH"):
        directories.extend(Path(path) for path in library_path.split(os.pathsep) if path)
    tvm_root = Path(__file__).resolve().parents[2]
    directories.extend((tvm_root / "build", tvm_root / "build" / "lib"))

    for directory in directories:
        candidates = (
            directory / "libtvm_compiler.so",
            directory / "lib" / "libtvm_compiler.so",
        )
        for candidate in candidates:
            if candidate.exists():
                _wrapper_library = ctypes.CDLL(str(candidate.resolve()), mode=ctypes.RTLD_GLOBAL)
                return _wrapper_library

    raise FileNotFoundError("Cannot find libtvm_compiler.so for custom datatype wrappers")


def _storage_bits(bits):
    for width in (8, 16, 32, 64):
        if bits <= width:
            return width
    raise ValueError(f"Unsupported custom bits={bits}, expected 1..64")


def _standard_converter(type_name, bits):
    cache_key = (type_name, bits)
    if cache_key in _converter_cache:
        return _converter_cache[cache_key]

    testcode_dir = Path(__file__).resolve().parents[1]
    if str(testcode_dir) not in sys.path:
        sys.path.insert(0, str(testcode_dir))

    from register import symbol_prefix_for_type_name

    prefix = symbol_prefix_for_type_name(type_name)
    from_name = f"FloatTo{prefix}_{bits}"
    to_name = f"{prefix}_{bits}ToFloat"
    library = _load_wrapper_library()
    missing = [name for name in (from_name, to_name) if not hasattr(library, name)]
    if missing:
        raise AttributeError(
            "Missing custom datatype wrapper symbols: " + ", ".join(missing)
        )

    storage_bits = _storage_bits(bits)
    storage_ctype = getattr(ctypes, f"c_uint{storage_bits}")
    storage_numpy_type = getattr(np, f"uint{storage_bits}")

    from_float = getattr(library, from_name)
    from_float.restype = storage_ctype
    from_float.argtypes = [ctypes.c_float]

    to_float = getattr(library, to_name)
    to_float.restype = ctypes.c_float
    to_float.argtypes = [storage_ctype]

    converter = {
        "from_float": np.vectorize(
            lambda value: from_float(float(value)), otypes=[storage_numpy_type]
        ),
        "to_float": np.vectorize(
            lambda value: to_float(storage_ctype(value)), otypes=[np.float32]
        ),
        "storage_dtype": f"uint{storage_bits}",
    }
    _converter_cache[cache_key] = converter
    return converter


def register_custom_datatypes(dtypes, target="llvm"):
    """Register custom families and standard wrapper lowering rules found in dtypes."""

    families = {}
    for dtype in dtypes:
        parsed = parse_custom_dtype(dtype)
        if parsed is not None:
            type_name, bits = parsed
            families.setdefault(type_name, set()).add(bits)

    if not families:
        return

    _load_wrapper_library()

    testcode_dir = Path(__file__).resolve().parents[1]
    if str(testcode_dir) not in sys.path:
        sys.path.insert(0, str(testcode_dir))

    from register import make_standard_profile, register, symbol_prefix_for_type_name
    from tvm.target.datatype import register_op
    from tvm.tirx.op import call_pure_extern

    assigned_codes = {}
    for type_name, bits_set in sorted(families.items()):
        type_code = custom_type_code(type_name)
        if previous_name := assigned_codes.get(type_code):
            raise ValueError(
                f"Custom datatype code collision: {previous_name} and {type_name} "
                f"both map to {type_code}"
            )
        assigned_codes[type_code] = type_name

        bits = tuple(sorted(bits_set))
        prefix = symbol_prefix_for_type_name(type_name)
        profile = make_standard_profile(type_name, bits=bits)
        profile = replace(
            profile,
            cast_from_float={(32, width): f"FloatTo{prefix}_{width}" for width in bits},
            cast_to_float={(width, 32): f"{prefix}_{width}ToFloat" for width in bits},
        )
        register(type_name, type_code, target=target, profile=profile)

        extra_cast_key = (type_name, target)
        if extra_cast_key not in _registered_extra_casts:
            extern_functions = {
                width: f"FloatTo{prefix}_{width}" for width in bits
            }
            to_float_functions = {
                width: f"{prefix}_{width}ToFloat" for width in bits
            }

            def lower_integer_cast(op, extern_functions=extern_functions):
                width = op.ty.dtype.bits
                if width not in extern_functions:
                    raise RuntimeError(f"Missing integer cast lowering for {op.ty}")
                lanes = op.ty.dtype.lanes
                float_dtype = "float32" if lanes == 1 else f"float32x{lanes}"
                storage_dtype = f"uint{width}" if lanes == 1 else f"uint{width}x{lanes}"
                value = tvm.tirx.Cast(float_dtype, op.value)
                return call_pure_extern(
                    storage_dtype, extern_functions[width], value
                )

            register_op(lower_integer_cast, "Cast", target, "int", type_name)
            register_op(lower_integer_cast, "Cast", target, "uint", type_name)

            def lower_custom_to_builtin(op, extern_functions=to_float_functions):
                width = op.value.ty.dtype.bits
                if width not in extern_functions:
                    raise RuntimeError(f"Missing builtin cast lowering for {op.value.ty}")
                lanes = op.value.ty.dtype.lanes
                float_dtype = "float32" if lanes == 1 else f"float32x{lanes}"
                value = call_pure_extern(
                    float_dtype, extern_functions[width], op.value
                )
                return tvm.tirx.Cast(op.ty, value)

            register_op(lower_custom_to_builtin, "Cast", target, type_name, "bool")
            register_op(lower_custom_to_builtin, "Cast", target, type_name, "int")
            register_op(lower_custom_to_builtin, "Cast", target, type_name, "uint")
            _registered_extra_casts.add(extra_cast_key)


def validate_dtype_arg(value):
    if value in ("float16", "float32"):
        return value

    parsed = parse_custom_dtype(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(
            f"Invalid dtype '{value}'. Expected float16/float32 or custom[name]bits"
        )

    _, bits = parsed
    if bits < 1 or bits > 64:
        raise argparse.ArgumentTypeError(f"Unsupported custom bits={bits}, expected 1..64")
    return value


def convert_float_array_to_runtime_tensor(array, dtype, device):
    """Create a runtime tensor using either a builtin or registered custom dtype."""

    parsed = parse_custom_dtype(dtype)
    if parsed is None:
        return tvm.runtime.tensor(np.asarray(array, dtype=dtype), device=device)

    type_name, bits = parsed
    converter = _standard_converter(type_name, bits)
    storage = converter["from_float"](np.asarray(array, dtype=np.float32))
    storage = np.asarray(storage, dtype=converter["storage_dtype"])
    custom_tensor = tvm.runtime.empty(storage.shape, dtype=dtype, device=device)
    tvm.runtime.tensor(storage, device=device).copyto(custom_tensor)
    return custom_tensor


def convert_runtime_tensor_to_float_numpy(tensor, dtype, device):
    """Convert a builtin or registered custom runtime tensor to float32 NumPy."""

    parsed = parse_custom_dtype(dtype)
    if parsed is None:
        return tensor.numpy()

    type_name, bits = parsed
    converter = _standard_converter(type_name, bits)
    storage_tensor = tvm.runtime.empty(
        tensor.shape, dtype=converter["storage_dtype"], device=device
    )
    tensor.copyto(storage_tensor)
    return converter["to_float"](storage_tensor.numpy())
