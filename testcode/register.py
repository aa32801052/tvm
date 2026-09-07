"""Uniform registration of custom datatypes and their lowering rules."""

import re
from dataclasses import dataclass
from typing import Mapping

from tvm.target.datatype import (
    create_lower_func,
    create_min_lower_func,
    lower_call_pure_extern,
    lower_ite,
    register as register_type,
    register_min_func,
    register_op,
)


@dataclass(frozen=True)
class LoweringProfile:
    """External functions used to lower one custom datatype family."""

    cast_from_float: Mapping[tuple[int, int], str]
    cast_to_float: Mapping[tuple[int, int], str]
    operations: Mapping[str, Mapping[int, str]]
    intrinsics: Mapping[str, Mapping[int, str]]
    float_immediates: Mapping[int, str]
    minimums: Mapping[int, str]
    lower_if_then_else: bool = True
    lower_call_pure_extern: bool = True


def symbol_prefix_for_type_name(type_name):
    """Convert a datatype name to the prefix used by its wrapper symbols."""

    name_parts = [part for part in re.split(r"[^A-Za-z0-9]+", type_name) if part]
    if not name_parts:
        raise ValueError("type_name must contain at least one letter or digit")
    return "".join(part[0].upper() + part[1:] for part in name_parts)


def make_standard_profile(
    type_name,
    *,
    bits=(32,),
    operations=("Add", "Sub", "Mul", "Div", "Max"),
    intrinsics=("Sqrt", "Exp", "Log", "Pow", "Sigmoid", "Tanh"),
):
    """Create lowering rules for the standard BYODT external-symbol convention.

    The external symbol prefix is derived from ``type_name``. For example,
    ``myfloat`` produces ``FloatToMyfloat_32``, ``Myfloat_32ToFloat``,
    ``Myfloat_32Add``, and ``MinMyfloat_32``.
    """

    symbol_prefix = symbol_prefix_for_type_name(type_name)

    bits = tuple(bits)
    if not bits:
        raise ValueError("bits cannot be empty")
    if any(not isinstance(num_bits, int) or num_bits <= 0 for num_bits in bits):
        raise ValueError("bits must contain positive integers")

    return LoweringProfile(
        cast_from_float={
            (num_bits, num_bits): f"FloatTo{symbol_prefix}_{num_bits}" for num_bits in bits
        },
        cast_to_float={
            (num_bits, num_bits): f"{symbol_prefix}_{num_bits}ToFloat" for num_bits in bits
        },
        operations={
            op_name: {
                num_bits: f"{symbol_prefix}_{num_bits}{op_name}" for num_bits in bits
            }
            for op_name in operations
        },
        intrinsics={
            f"tirx.{intrinsic_name.lower()}": {
                num_bits: f"{symbol_prefix}_{num_bits}{intrinsic_name}" for num_bits in bits
            }
            for intrinsic_name in intrinsics
        },
        float_immediates={
            num_bits: f"FloatTo{symbol_prefix}_{num_bits}" for num_bits in bits
        },
        minimums={num_bits: f"Min{symbol_prefix}_{num_bits}" for num_bits in bits},
    )


_LOWERING_PROFILES = {}
_REGISTERED_TYPE_CODES = {}
_REGISTERED_LOWERINGS = set()


def register_lowering_profile(type_name, profile):
    """Install the lowering profile used by subsequent registrations."""

    if not isinstance(profile, LoweringProfile):
        raise TypeError("profile must be a LoweringProfile")
    if any(name == type_name for name, _target in _REGISTERED_LOWERINGS):
        raise RuntimeError(f"cannot replace lowering profile for registered type {type_name}")
    _LOWERING_PROFILES[type_name] = profile


def _register_lowering_rules(type_name, target, profile):
    if profile.cast_from_float:
        register_op(
            create_lower_func(profile.cast_from_float),
            "Cast",
            target,
            "float",
            type_name,
        )
    if profile.cast_to_float:
        register_op(
            create_lower_func(profile.cast_to_float),
            "Cast",
            target,
            type_name,
            "float",
        )

    for op_name, extern_functions in profile.operations.items():
        register_op(
            create_lower_func(extern_functions),
            op_name,
            target,
            type_name,
        )

    for intrinsic_name, extern_functions in profile.intrinsics.items():
        register_op(
            create_lower_func(extern_functions),
            "Call",
            target,
            type_name,
            intrinsic_name=intrinsic_name,
        )

    if profile.float_immediates:
        register_op(
            create_lower_func(profile.float_immediates),
            "FloatImm",
            target,
            type_name,
        )
    if profile.lower_if_then_else:
        register_op(lower_ite, "Call", target, type_name, intrinsic_name="tirx.if_then_else")
    if profile.lower_call_pure_extern:
        register_op(
            lower_call_pure_extern,
            "Call",
            target,
            type_name,
            intrinsic_name="tirx.call_pure_extern",
        )
    if profile.minimums:
        register_min_func(create_min_lower_func(profile.minimums, type_name), type_name)


def register(type_name, type_code, *, target="llvm", profile=None):
    """Register a custom datatype and its lowering rules through one interface.

    If ``profile`` is omitted, a profile previously installed for ``type_name``
    is used. Otherwise, a standard profile derived from ``type_name`` is used.
    """

    previous_code = _REGISTERED_TYPE_CODES.get(type_name)
    if previous_code is not None and previous_code != type_code:
        raise ValueError(
            f"datatype {type_name} is already registered with code {previous_code}, "
            f"not {type_code}"
        )

    if previous_code is None:
        register_type(type_name, type_code)
        _REGISTERED_TYPE_CODES[type_name] = type_code

    lowering_key = (type_name, target)
    if lowering_key in _REGISTERED_LOWERINGS:
        return

    selected_profile = profile or _LOWERING_PROFILES.get(type_name)
    if selected_profile is None:
        selected_profile = make_standard_profile(type_name)
    if not isinstance(selected_profile, LoweringProfile):
        raise TypeError("profile must be a LoweringProfile")

    _register_lowering_rules(type_name, target, selected_profile)
    _REGISTERED_LOWERINGS.add(lowering_key)


__all__ = [
    "LoweringProfile",
    "make_standard_profile",
    "register",
    "register_lowering_profile",
    "symbol_prefix_for_type_name",
]
