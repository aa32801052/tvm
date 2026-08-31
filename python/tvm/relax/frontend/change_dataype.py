# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Change datatype pass for Relax."""

from tvm.ir import Call, PrimType, TupleType

from ..expr import Constant, DataTypeImm, Expr, Var
from ..expr_functor import PyExprMutator, mutator
from ..op import astype
from ..type import TensorType
from ..transform.transform import function_pass


@mutator
class _ChangeDatatypeMutator(PyExprMutator):
    """Rewrite Relax expressions from one datatype to another."""

    def __init__(self, src, dst, mod):
        super().__init__(mod)
        self.src = PrimType(src)
        self.dst = PrimType(dst)

    def _rewrite_type(self, ty):
        if isinstance(ty, TensorType) and ty.dtype == self.src:
            if ty.shape is not None:
                return TensorType(ty.shape, self.dst, ty.vdevice, span=ty.span)
            return TensorType(dtype=self.dst, vdevice=ty.vdevice, ndim=ty.ndim, span=ty.span)

        if isinstance(ty, TupleType):
            new_fields = [self._rewrite_type(field) for field in ty.fields]
            if any(not old.same_as(new) for old, new in zip(ty.fields, new_fields)):
                return TupleType(new_fields, ty.span)

        return ty

    def _rewrite_attrs(self, attrs):
        if attrs is None:
            return attrs

        fields = attrs.__class__.__tvm_ffi_type_info__.fields
        values = {field.name: getattr(attrs, field.name) for field in fields}
        changed = False
        for name in ("dtype", "out_dtype"):
            if name in values and values[name] == self.src.dtype:
                values[name] = self.dst.dtype
                changed = True
        return type(attrs)(**values) if changed else attrs

    def visit_var_def_(self, var: Var) -> Var:
        new_ty = self._rewrite_type(var.ty)
        if new_ty.same_as(var.ty):
            return var
        return type(var)(var.name, new_ty, var.span)

    def visit_constant_(self, op: Constant) -> Expr:
        if op.data.dtype == self.src.dtype:
            return astype(op, self.dst)
        return op

    def visit_data_type_imm_(self, op: DataTypeImm) -> Expr:
        if op.value == self.src.dtype:
            return DataTypeImm(self.dst.dtype, op.span)
        return op

    def visit_call_(self, op: Call) -> Expr:
        new_op = self.visit_expr(op.op)
        new_args = [self.visit_expr(arg) for arg in op.args]
        new_attrs = self._rewrite_attrs(op.attrs)
        new_ty_args = [self._rewrite_type(ty_arg) for ty_arg in op.ty_args]
        new_ret_ty = self._rewrite_type(op.ty)
        return Call(new_op, new_args, new_attrs, new_ty_args, op.span, ret_ty=new_ret_ty)


@function_pass(opt_level=0)
class ChangeDatatype:
    """Change all Relax tensor occurrences of ``src`` to ``dst``.

    Parameters
    ----------
    src : str
        The source datatype, for example ``"float32"``.

    dst : str
        The destination datatype, for example ``"custom[myfloat]32"``.
    """

    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def transform_function(self, func, mod, ctx):
        return _ChangeDatatypeMutator(self.src, self.dst, mod).visit_expr(func)
