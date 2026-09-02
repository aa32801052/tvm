"""Datatype-neutral registration and tensor conversion helpers."""

import argparse
import ctypes
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm

_CUSTOM_DTYPE_RE = re.compile(r"^custom\[([A-Za-z_][A-Za-z0-9_]*)\](\d+)$")
_REGISTRATION_ENV = "TVM_CUSTOM_DATATYPE_CONFIG"

_libtvm_handle = None
_converter_cache = {}
_custom_converter_factories = []


@dataclass(frozen=True)
class CustomDTypeSpec:
    name: str
    bits: int

    @property
    def dtype(self):
        return f"custom[{self.name}]{self.bits}"


def parse_custom_dtype(dtype) -> CustomDTypeSpec | None:
    if dtype is None:
        return None
    match = _CUSTOM_DTYPE_RE.fullmatch(str(dtype))
    if match is None:
        return None
    return CustomDTypeSpec(name=match.group(1), bits=int(match.group(2)))


def is_custom_dtype(dtype) -> bool:
    return parse_custom_dtype(dtype) is not None


def normalize_dtype_suffix(dtype) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(dtype)).strip("_")


def validate_dtype_arg(value: str) -> str:
    spec = parse_custom_dtype(value)
    if spec is not None:
        if not 1 <= spec.bits <= 64:
            raise argparse.ArgumentTypeError(
                f"Unsupported custom bits={spec.bits}, expected 1..64"
            )
        return value

    try:
        tvm.ir.PrimType(value)
    except (TypeError, ValueError, tvm.error.TVMError) as err:
        raise argparse.ArgumentTypeError(f"Invalid dtype {value!r}") from err
    return value


def _registration_api():
    testcode_dir = Path(__file__).resolve().parent.parent
    if str(testcode_dir) not in sys.path:
        sys.path.insert(0, str(testcode_dir))

    from register import make_standard_profile, register, symbol_prefix_for_type_name

    return make_standard_profile, register, symbol_prefix_for_type_name


def _type_code_for_name(type_name: str) -> int:
    """Derive a stable custom type code from a datatype name."""

    digest = hashlib.blake2s(type_name.encode("utf-8"), digest_size=2).digest()
    return 129 + int.from_bytes(digest, byteorder="little") % 127


def configure_custom_datatypes(dtypes: Iterable[str], target="llvm"):
    """Configure and register the custom datatype used by compiler workers."""

    specs = {spec for dtype in dtypes if (spec := parse_custom_dtype(dtype)) is not None}
    names = {spec.name for spec in specs}
    if not names:
        os.environ.pop(_REGISTRATION_ENV, None)
        return
    if len(names) != 1:
        raise ValueError(
            "A single compilation currently supports one custom datatype family; "
            f"received {sorted(names)}"
        )
    type_name = names.pop()

    config = {
        "name": type_name,
        "bits": sorted(spec.bits for spec in specs),
        "type_code": _type_code_for_name(type_name),
        "target": target,
    }
    os.environ[_REGISTRATION_ENV] = json.dumps(config)
    register_custom_datatypes()


def register_custom_datatypes():
    """Worker initializer that registers the configured datatype and lowerings."""

    encoded_config = os.environ.get(_REGISTRATION_ENV)
    if not encoded_config:
        return
    config = json.loads(encoded_config)
    make_standard_profile, register, _ = _registration_api()
    profile = make_standard_profile(config["name"], bits=config["bits"])
    register(
        config["name"],
        int(config["type_code"]),
        target=config["target"],
        profile=profile,
    )


def _storage_bits_for_width(bits: int) -> int:
    for storage_bits in (8, 16, 32, 64):
        if bits <= storage_bits:
            return storage_bits
    raise ValueError(f"unsupported custom datatype width: {bits}")


def _numpy_dtype_for_storage_bits(storage_bits: int) -> str:
    return f"uint{storage_bits}"


def _numpy_scalar_type_for_storage_bits(storage_bits: int):
    return getattr(np, f"uint{storage_bits}")


def _ctypes_uint_for_storage_bits(storage_bits: int):
    return getattr(ctypes, f"c_uint{storage_bits}")


def _library_candidates():
    filenames = ("libtvm_compiler.so", "libtvm.so")
    directories = []
    if env_path := os.environ.get("TVM_LIBRARY_PATH"):
        directories.extend(Path(path) for path in env_path.split(os.pathsep) if path)

    tvm_root = Path(__file__).resolve().parents[2]
    directories.extend((tvm_root / "build" / "lib", tvm_root / "build"))

    for directory in directories:
        for candidate_dir in (directory, directory / "lib"):
            for filename in filenames:
                yield candidate_dir / filename


def _get_libtvm_handle():
    global _libtvm_handle
    if _libtvm_handle is not None:
        return _libtvm_handle

    checked = []
    for path in _library_candidates():
        path = path.resolve()
        if path in checked:
            continue
        checked.append(path)
        if path.exists():
            _libtvm_handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            return _libtvm_handle

    raise FileNotFoundError(
        "Cannot find a TVM shared library. Checked: " + ", ".join(map(str, checked))
    )


def register_custom_converter_factory(factory: Callable[[CustomDTypeSpec], dict | None]):
    """Register a converter factory, checked before the standard wrapper ABI."""

    if not callable(factory):
        raise TypeError("factory must be callable")
    _custom_converter_factories.append(factory)
    return factory


def register_custom_converter_template(
    name_pattern,
    from_float_core,
    to_float_core,
    *,
    min_bits=1,
    max_bits=64,
):
    """Register NumPy conversion functions for a custom datatype name pattern."""

    if min_bits < 1 or max_bits > 64 or min_bits > max_bits:
        raise ValueError("invalid bits range for custom converter")
    pattern = re.compile(name_pattern)

    def factory(spec):
        match = pattern.fullmatch(spec.name)
        if match is None:
            return None
        if not min_bits <= spec.bits <= max_bits:
            raise ValueError(
                f"Unsupported {spec.name} bits={spec.bits}, expected {min_bits}..{max_bits}"
            )

        storage_bits = _storage_bits_for_width(spec.bits)
        storage_dtype = _numpy_dtype_for_storage_bits(storage_bits)

        def from_float(array):
            result = from_float_core(np.asarray(array, dtype=np.float32), spec.bits, match)
            return np.asarray(result).astype(storage_dtype, copy=False)

        def to_float(array):
            storage = np.asarray(array).astype(storage_dtype, copy=False)
            return np.asarray(to_float_core(storage, spec.bits, match), dtype=np.float32)

        return {
            "from_float": from_float,
            "to_float": to_float,
            "storage_bits": storage_bits,
        }

    return register_custom_converter_factory(factory)


def _standard_converter(spec):
    cache_key = (spec.name, spec.bits)
    if cache_key in _converter_cache:
        return _converter_cache[cache_key]

    _, _, symbol_prefix_for_type_name = _registration_api()
    prefix = symbol_prefix_for_type_name(spec.name)
    from_name = f"FloatTo{prefix}_{spec.bits}"
    to_name = f"{prefix}_{spec.bits}ToFloat"
    library = _get_libtvm_handle()

    missing = [name for name in (from_name, to_name) if not hasattr(library, name)]
    if missing:
        raise AttributeError(f"missing custom datatype wrapper symbols: {', '.join(missing)}")

    storage_bits = _storage_bits_for_width(spec.bits)
    storage_ctype = _ctypes_uint_for_storage_bits(storage_bits)
    storage_numpy_type = _numpy_scalar_type_for_storage_bits(storage_bits)

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
        "storage_bits": storage_bits,
    }
    _converter_cache[cache_key] = converter
    return converter


def _resolve_converter(spec):
    for factory in _custom_converter_factories:
        if converter := factory(spec):
            return converter
    return _standard_converter(spec)


def storage_numpy_dtype_for_custom(dtype) -> str:
    spec = parse_custom_dtype(dtype)
    if spec is None:
        raise ValueError(f"{dtype} is not a custom dtype")
    return _numpy_dtype_for_storage_bits(_storage_bits_for_width(spec.bits))


def unpack_storage_numpy_to_custom_tensor(storage, dtype, device):
    storage_dtype = storage_numpy_dtype_for_custom(dtype)
    storage = np.asarray(storage).astype(storage_dtype, copy=False)
    custom_tensor = tvm.runtime.empty(storage.shape, dtype=dtype, device=device)
    tvm.runtime.tensor(storage, device=device).copyto(custom_tensor)
    return custom_tensor


def pack_custom_tensor_to_storage_numpy(custom_tensor, dtype, device):
    storage_dtype = storage_numpy_dtype_for_custom(dtype)
    storage_tensor = tvm.runtime.empty(custom_tensor.shape, dtype=storage_dtype, device=device)
    custom_tensor.copyto(storage_tensor)
    return storage_tensor.numpy()


def convert_float_array_to_custom_tensor(array, dtype, device):
    spec = parse_custom_dtype(dtype)
    if spec is None:
        raise ValueError(f"{dtype} is not a custom dtype")
    storage = _resolve_converter(spec)["from_float"](np.asarray(array, dtype=np.float32))
    return unpack_storage_numpy_to_custom_tensor(storage, dtype, device)


def convert_custom_tensor_to_float_numpy(custom_tensor, dtype, device):
    spec = parse_custom_dtype(dtype)
    if spec is None:
        return custom_tensor.numpy()
    storage = pack_custom_tensor_to_storage_numpy(custom_tensor, dtype, device)
    return _resolve_converter(spec)["to_float"](storage)
