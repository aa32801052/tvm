/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/transform/to_mixed_precision_custom.cc
 * \brief Automatic mixed precision pass with custom datatype support.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>

#include "../../target/datatype/registry.h"
#include "../op/nn/convolution.h"
#include "../op/tensor/datatype.h"
#include "../op/tensor/linear_algebra.h"
#include "infer_amp_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Get the mixed precision policy for a call node.
 *
 * Returns the TMixedPrecisionPolicy for the operation, or -1 if not an op call.
 */
static int GetMixedPrecisionInfo(const CallNode* call_node) {
  const OpNode* op_node = call_node->op.as<OpNode>();
  if (op_node == nullptr) {
    return -1;
  }
  Op op = ffi::GetRef<Op>(op_node);
  auto attr_map = Op::GetAttrMap<TMixedPrecisionPolicy>("TMixedPrecisionPolicy");
  return attr_map.count(op) ? attr_map[op] : MixedPrecisionPolicyKind::kNever;
}

/*!
 * \brief Custom dtype configuration for mixed precision.
 */
struct CustomMixedPrecisionConfig {
  DLDataType src_dtype;
  DLDataType dst_dtype;
  DLDataType acc_dtype;
};

static bool IsNumericallySensitiveOp(const Op& op) {
  static const std::unordered_set<std::string> kSensitiveOps = {
      "relax.nn.softmax",      "relax.nn.log_softmax", "relax.nn.layer_norm",
      "relax.nn.group_norm",   "relax.nn.instance_norm", "relax.nn.rms_norm",
      "relax.sum",             "relax.mean",           "relax.variance",
      "relax.std",             "relax.prod",           "relax.max",
      "relax.min",             "relax.cumsum",         "relax.cumprod",
      "relax.exp",             "relax.log",            "relax.power",
      "relax.sqrt",            "relax.rsqrt",          "relax.equal",
      "relax.greater",         "relax.greater_equal",  "relax.less",
      "relax.less_equal",      "relax.not_equal",
  };
  return kSensitiveOps.count(std::string(op->name)) != 0;
}

static bool IsPolicyFollowOverrideOp(const Op& op) {
  static const std::unordered_set<std::string> kFollowOps = {
      "relax.where",
  };
  return kFollowOps.count(std::string(op->name)) != 0;
}

static bool IsLayoutOrShapeFollowOp(const Op& op) {
  static const std::unordered_set<std::string> kLayoutOps = {
      "relax.reshape",      "relax.expand_dims", "relax.permute_dims",
      "relax.broadcast_to", "relax.squeeze",     "relax.flatten",
      "relax.concat",       "relax.split",       "relax.layout_transform",
  };
  return kLayoutOps.count(std::string(op->name)) != 0;
}

static int ApplyCustomMixedPrecisionPolicyOverrides(const Op& op, int policy) {
  if (IsNumericallySensitiveOp(op)) {
    return kNever;
  }
  if (IsPolicyFollowOverrideOp(op)) {
    return kFollow;
  }
  return policy;
}

/*!
 * \brief Validate dtype combination for mixed precision
 * \param src Source custom dtype
 * \param dst Destination custom dtype
 * \param acc Accumulator custom dtype
 * \throws Error if invalid combination
 */
static void ValidateMixedPrecisionDTypes(DLDataType src, DLDataType dst, DLDataType acc) {
  auto* registry = datatype::Registry::Global();
  auto is_registered = [registry](DLDataType dtype) {
    return registry->GetTypeRegistered(static_cast<uint8_t>(dtype.code));
  };

  TVM_FFI_ICHECK(is_registered(src) && is_registered(dst) && is_registered(acc))
      << "Custom mixed precision requires registered custom datatypes";
  TVM_FFI_ICHECK_EQ(src.code, dst.code)
      << "Source and destination must belong to the same custom datatype family";
  TVM_FFI_ICHECK_EQ(src.code, acc.code)
      << "Source and accumulator must belong to the same custom datatype family";
  TVM_FFI_ICHECK_EQ(src.lanes, 1) << "Custom mixed precision only supports scalar lanes";
  TVM_FFI_ICHECK_EQ(dst.lanes, 1) << "Custom mixed precision only supports scalar lanes";
  TVM_FFI_ICHECK_EQ(acc.lanes, 1) << "Custom mixed precision only supports scalar lanes";
  TVM_FFI_ICHECK_LE(dst.bits, src.bits)
      << "Destination dtype must not have more bits than source dtype";
  TVM_FFI_ICHECK_GE(acc.bits, dst.bits)
      << "Accumulator dtype must have at least as many bits as destination dtype";
}

/*!
 * \brief DType decision collector with configurable source/destination dtypes.
 */
class CustomDTypeDecisionCollector : public ExprVisitor {
 public:
  explicit CustomDTypeDecisionCollector(const CustomMixedPrecisionConfig& config)
      : src_dtype_(config.src_dtype),
        dst_dtype_(config.dst_dtype) {}

  static VarDTypeMap Collect(Function func, const CustomMixedPrecisionConfig& config) {
    CustomDTypeDecisionCollector collector(config);
    collector.VisitExpr(func);
    return std::move(collector.dtype_map_);
  }

 private:
  NType GetDType(const Var& var) {
    auto it = dtype_map_.find(var);
    if (it == dtype_map_.end()) {
      return NTypeUnknown(GetType(var));
    }
    return it->second;
  }

  NType NTypeUnknown(const Type& ty) {
    auto fmapleaf = [](const Type&) -> NType { return NType(ffi::String("")); };
    return MapToNestedMsg<ffi::String>(ty, fmapleaf);
  }

  void UpdateVarDTypeMap(const Var& var, const NType& dtype) {
    auto it = dtype_map_.find(var);
    if (it == dtype_map_.end()) {
      dtype_map_[var] = dtype;
    } else {
      dtype_map_[var] = NTypeMerge(it->second, dtype);
    }
  }

  void RequireArgsToType(ffi::Array<Expr> args, ffi::Array<NType> to) {
    TVM_FFI_ICHECK(args.size() == to.size()) << "Invalid target dtypes";
    for (size_t i = 0; i < args.size(); ++i) {
      auto fvisitleaf = [&](const Expr& expr, NType to) {
        if (const auto* var = expr.as<VarNode>()) {
          UpdateVarDTypeMap(ffi::GetRef<Var>(var), to);
        } else if (expr->IsInstance<ConstantNode>()) {
          return;
        } else {
          TVM_FFI_THROW(InternalError) << "Unsupported argument type: " << expr->GetTypeKey();
        }
      };
      DecomposeNestedMsg(args[i], to[i], fvisitleaf);
    }
  }

  void RequireArgsToType(ffi::Array<Expr> args, DLDataType to) {
    std::vector<Expr> arg_arr;
    std::vector<NType> to_arr;
    for (const Expr& arg : args) {
      if (IsNestedTensor(arg)) {
        arg_arr.push_back(arg);
        to_arr.push_back(NTypeFrom(arg, to));
      }
    }
    RequireArgsToType(std::move(arg_arr), std::move(to_arr));
  }

  void RequireArgsToFollowOutput(ffi::Array<Expr> args, const NType& output_type) {
    ffi::String required("");
    ForEachLeaf<ffi::String>(output_type, [&](const ffi::String& dtype) {
      if (dtype == "") {
        return;
      }
      if (required == "") {
        required = dtype;
      } else {
        required = NTypeMerge(NType(required), NType(dtype)).LeafValue();
      }
    });

    std::vector<Expr> arg_arr;
    std::vector<NType> to_arr;
    for (const Expr& arg : args) {
      if (!IsNestedTensor(arg)) {
        continue;
      }
      arg_arr.push_back(arg);
      if (required == "") {
        to_arr.push_back(NTypeUnknown(GetType(arg)));
      } else {
        to_arr.push_back(NTypeFrom(arg, ffi::StringToDLDataType(required)));
      }
    }
    RequireArgsToType(std::move(arg_arr), std::move(to_arr));
  }

  void VisitVars_(const VarNode* op) {
    Var var = ffi::GetRef<Var>(op);
    if (IsNestedTensor(var)) {
      UpdateVarDTypeMap(var, NTypeFrom(var, src_dtype_));
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) final { VisitVars_(op); }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    auto policy = GetMixedPrecisionInfo(call_node);
    if (policy == -1) {
      ExprVisitor::VisitBinding_(binding, call_node);
      return;
    }
    const auto* op_node = call_node->op.as<OpNode>();
    TVM_FFI_ICHECK(op_node != nullptr);
    Op op = ffi::GetRef<Op>(op_node);
    policy = ApplyCustomMixedPrecisionPolicyOverrides(op, policy);
    if (policy == kAlways) {
      RequireArgsToType(call_node->args, dst_dtype_);
    } else if (policy == kFollow) {
      if (IsLayoutOrShapeFollowOp(op)) {
        RequireArgsToFollowOutput(call_node->args, GetDType(binding->var));
      } else {
        RequireArgsToType(call_node->args, src_dtype_);
      }
    } else if (policy == kNever) {
      RequireArgsToType(call_node->args, src_dtype_);
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple_node) final {
    NType lhs_type = GetDType(binding->var);
    RequireArgsToType(tuple_node->fields, lhs_type.NestedArray());
  }

  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item_node) final {
    NType lhs_type = GetDType(binding->var);

    const TupleTypeNode* ty = tuple_get_item_node->tuple->ty.as<TupleTypeNode>();
    TVM_FFI_ICHECK(ty != nullptr) << "TupleGetItemNode must have TupleType";
    const VarNode* tuple_var = tuple_get_item_node->tuple.as<VarNode>();
    bool tuple_has_type = false;
    if (tuple_var != nullptr) {
      tuple_has_type = dtype_map_.count(ffi::GetRef<Var>(tuple_var)) != 0;
    }
    std::vector<NType> require_rhs;
    for (size_t i = 0; i < ty->fields.size(); ++i) {
      if (i == static_cast<size_t>(tuple_get_item_node->index)) {
        require_rhs.push_back(lhs_type);
      } else {
        if (tuple_has_type) {
          NType existing_type = dtype_map_.at(ffi::GetRef<Var>(tuple_var));
          if (!existing_type.IsLeaf()) {
            ffi::Array<NType> existing_arr = existing_type.NestedArray();
            if (i < existing_arr.size()) {
              require_rhs.push_back(existing_arr[i]);
              continue;
            }
          }
        }
        require_rhs.push_back(NTypeUnknown(ty->fields[i]));
      }
    }
    RequireArgsToType({tuple_get_item_node->tuple}, {NType(require_rhs)});
  }

  void VisitExpr_(const SeqExprNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->body);
    for (auto it = op->blocks.rbegin(); it != op->blocks.rend(); it++) {
      this->VisitBindingBlock(*it);
    }
    if (auto* ty = op->ty.as<TypeNode>()) {
      this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
    }
  }

  void VisitBindingBlock_(const BindingBlockNode* block) { return; }

  void VisitBindingBlock_(const DataflowBlockNode* block) {
    for (auto it = block->bindings.rbegin(); it != block->bindings.rend(); it++) {
      this->VisitBinding(*it);
    }
  }

  void VisitExpr_(const IfNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->true_branch);
    this->VisitExpr(op->false_branch);
    this->VisitExpr(op->cond);
    if (auto* ty = op->ty.as<TypeNode>()) {
      this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
    }
  }

  DLDataType src_dtype_;
  DLDataType dst_dtype_;
  VarDTypeMap dtype_map_;
};

// Forward declaration of ToMixedPrecisionCustom
Expr ToMixedPrecisionCustom(const Function& f, const CustomMixedPrecisionConfig& config,
                            ffi::Optional<ffi::Array<ffi::String>> dst_input_names);

namespace transform {

/*!
 * \brief Create mixed precision pass with custom datatype support.
 *
 * \param src_dtype Source custom datatype
 * \param dst_dtype Destination custom datatype used for computation
 * \param acc_dtype Accumulator custom datatype used by gemm/conv operators
 * \param dst_input_names Optional list of parameter names to convert to dst_dtype
 * \return The transform pass
 */
Pass ToMixedPrecisionCustom(DLDataType src_dtype, DLDataType dst_dtype, DLDataType acc_dtype,
                            ffi::Optional<ffi::Array<ffi::String>> dst_input_names) {
  ValidateMixedPrecisionDTypes(src_dtype, dst_dtype, acc_dtype);
  CustomMixedPrecisionConfig config{src_dtype, dst_dtype, acc_dtype};
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return ToMixedPrecisionCustom(f, config, dst_input_names).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, 0, "ToMixedPrecisionCustom", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ToMixedPrecisionCustom", ToMixedPrecisionCustom);
}

}  // namespace transform

/*!
 * \brief ToMixedPrecisionCustomRewriter with configurable datatypes.
 */
class ToMixedPrecisionCustomRewriter : public ExprMutator {
 public:
  explicit ToMixedPrecisionCustomRewriter(
      const VarDTypeMap* only_dst_map,
      const CustomMixedPrecisionConfig& config,
      const std::unordered_set<std::string>& dst_input_names)
      : only_dst_map_(only_dst_map),
        config_(config),
        dst_input_names_(dst_input_names) {}

 private:
  Var GetRemapped(const Var& var) {
    auto it = var_remap_.find(var);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      if (dst_input_names_.count(var->name)) {
        auto ty = GetType(var);
        if (auto tensor_ty = ty.as<TensorTypeNode>()) {
          VDevice vdev = VDevice();
          if (tensor_ty->vdevice.has_value()) {
            vdev = tensor_ty->vdevice.value();
          }
          TensorType dst_ty(tensor_ty->shape.value(), PrimType(config_.dst_dtype), vdev,
                            tensor_ty->span);
          Var dst_var(var->name, dst_ty, var->span);
          var_remap_[var] = dst_var;
          return dst_var;
        }
      }
      return var;
    }
  }

  ffi::Array<Expr> RemapArgs(const ffi::Array<Expr>& args) {
    return args.Map([this](Expr arg) { return VarReplacer::Replace(arg, var_remap_); });
  }

  // Rewrite expr to given dtype
  Expr RewriteExpr(const Expr& expr, const NType& to) {
    auto fvisitleaf = [&](const Expr& expr, std::array<NType, 1> to) -> Expr {
      const auto* tensor = GetTypeAs<TensorTypeNode>(expr);
      TVM_FFI_ICHECK(tensor != nullptr) << "Only support rewriting tensor expr";
      if (NTypeEqual()(to[0], NTypeFrom(expr))) return expr;
      if (tensor->IsUnknownDtype()) return expr;
      DLDataType tensor_dtype = tensor->dtype.value()->dtype;
      if (tensor_dtype != config_.src_dtype && tensor_dtype != config_.dst_dtype &&
          tensor_dtype != config_.acc_dtype) {
        return expr;
      }
      return astype(expr, ffi::StringToDLDataType(to[0].LeafValue()));
    };
    return TransformTupleLeaf<ffi::String>(expr, std::array<NType, 1>({to}), fvisitleaf);
  }

  ffi::Array<Expr> RewriteArgs(const ffi::Array<Expr>& args, DLDataType to) {
    ffi::Array<Expr> new_args;
    for (const Expr& arg : args) {
      if (IsNestedTensor(arg)) {
        new_args.push_back(RewriteExpr(arg, NTypeFrom(arg, to)));
      } else {
        new_args.push_back(arg);
      }
    }
    return new_args;
  }

  // Check if all arguments can be cast to dst_dtype
  bool AllDstCastable(const ffi::Array<Expr>& args) {
    auto is_dst = [this](Type ty) {
      if (auto tensor_ty = ty.as<TensorTypeNode>();
          tensor_ty && tensor_ty->dtype == PrimType(config_.dst_dtype)) {
        return true;
      }
      return false;
    };

    auto is_in_dst_range = [this](const ConstantNode* constant) {
      const auto& data = constant->data;
      if (data->dtype.lanes > 1) {
        return false;
      }

      if (data.DataType() == config_.dst_dtype) {
        return true;
      }

      // For custom types, we assume they can be cast
      // (proper range checking would require custom dtype-specific logic)
      return true;
    };

    for (const Expr& arg : args) {
      auto ty = GetType(arg);
      auto constant = arg.as<ConstantNode>();
      auto tuple = arg.as<TupleNode>();
      auto tensor_ty = ty.as<TensorTypeNode>();

      if (!IsNestedTensor(arg) || is_dst(ty) ||
          (tensor_ty && !tensor_ty->IsUnknownDtype() &&
           tensor_ty->dtype != PrimType(config_.src_dtype) &&
           tensor_ty->dtype != PrimType(config_.dst_dtype) &&
           tensor_ty->dtype != PrimType(config_.acc_dtype)) ||
          (constant && is_in_dst_range(constant)) ||
          (tuple && AllDstCastable(tuple->fields))) {
        continue;
      } else {
        return false;
      }
    }

    return true;
  }

  bool ShouldStoreBindingAsDst(const Var& var) {
    auto it = only_dst_map_->find(var);
    if (it == only_dst_map_->end()) {
      return false;
    }
    const std::string dst_dtype_str = ffi::DLDataTypeToString(config_.dst_dtype);
    bool has_leaf = false;
    bool only_dst = true;
    ForEachLeaf<ffi::String>(it->second, [&](const ffi::String& dtype) {
      if (dtype == "") {
        return;
      }
      has_leaf = true;
      if (dtype != dst_dtype_str) {
        only_dst = false;
      }
    });
    return has_leaf && only_dst;
  }

  void CastIfDstOnly(const Var& var) {
    TVM_FFI_ICHECK(builder_->CurrentBlockIsDataFlow());
    Var cur_var = GetRemapped(var);

    auto it = only_dst_map_->find(var);
    if (it == only_dst_map_->end()) return;

    // Get the to dtype
    auto dst_dtype_str = ffi::DLDataTypeToString(config_.dst_dtype);
    auto fcombine = [&dst_dtype_str](const ffi::String& from,
                                     const ffi::String& required) -> ffi::String {
      return required == dst_dtype_str ? required : from;
    };
    NType from = NTypeFrom(cur_var);
    NType to = CombineNestedMsg<ffi::String>(from, it->second, fcombine);
    Expr rewrite = RewriteExpr(cur_var, to);

    if (!rewrite.same_as(cur_var)) {
      var_remap_[var] = builder_->Emit(rewrite);
    }
  }

  Expr VisitVar_(const Var& var) {
    auto it = var_remap_.find(var);
    if (it != var_remap_.end()) {
      return RewriteExpr(it->second, NTypeFrom(var));
    }
    return var;
  }

  Expr VisitExpr_(const VarNode* op) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      return ExprMutator::VisitExpr_(op);
    }
    return VisitVar_(ffi::GetRef<Var>(op));
  }

  Var VisitVarDef(const Var& var) { return GetRemapped(var); }

  void VisitBinding(const Binding& binding) {
    ExprMutator::VisitBinding(binding);
    if (!builder_->CurrentBlockIsDataFlow()) return;
    CastIfDstOnly(binding->var);
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      ExprMutator::VisitBinding_(binding, call_node);
      return;
    }

    auto policy = GetMixedPrecisionInfo(call_node);
    if (policy == -1) {
      ExprMutator::VisitBinding_(binding, call_node);
      return;
    }

    const auto* op_node = call_node->op.as<OpNode>();
    TVM_FFI_ICHECK(op_node != nullptr);
    Op op = ffi::GetRef<Op>(op_node);
    policy = ApplyCustomMixedPrecisionPolicyOverrides(op, policy);
    if (wrap_param_op.same_as(op)) {
      ReEmitBinding(binding, call_node->args[0]);
      return;
    }

    Call new_call = ffi::GetRef<Call>(call_node);
    new_call.CopyOnWrite()->args = RemapArgs(new_call->args);

    std::optional<DLDataType> opt_new_dtype = std::nullopt;

    if (policy == kAlways) {
      // Use dst_dtype for computation, acc_dtype for output
      opt_new_dtype = config_.dst_dtype;
      auto attr_map = Op::GetAttrMap<FInferMixedPrecision>("FInferMixedPrecision");
      TVM_FFI_ICHECK(attr_map.count(op));
      new_call = attr_map[op](new_call, config_.acc_dtype);
    } else if (policy == kFollow) {
      opt_new_dtype = AllDstCastable(new_call->args) ? config_.dst_dtype : config_.src_dtype;
    } else if (policy == kNever) {
      if (!new_call->args.same_as(call_node->args)) {
        ffi::Array<Expr> new_typed_args;
        for (size_t i = 0; i < call_node->args.size(); i++) {
          auto arg = new_call->args[i];
          auto old_ntype = NTypeFrom(call_node->args[i]);
          new_typed_args.push_back(RewriteExpr(arg, old_ntype));
        }
        new_call.CopyOnWrite()->args = new_typed_args;
      }
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }

    Expr new_value = new_call;
    if (opt_new_dtype) {
      auto new_dtype = opt_new_dtype.value();
      new_call.CopyOnWrite()->args = RewriteArgs(new_call->args, new_dtype);
      new_call.CopyOnWrite()->ty = Type::Missing();

      new_value = builder_->Normalize(Call(new_call));

      if (!binding->var->IsInstance<DataflowVarNode>()) {
        new_value = RewriteExpr(new_value, NTypeFrom(binding->var));
      } else if (policy == kAlways && binding->var->IsInstance<DataflowVarNode>()) {
        if (ShouldStoreBindingAsDst(binding->var)) {
          new_value = RewriteExpr(new_value, NTypeFrom(new_value, new_dtype));
        }
      }
    }

    ReEmitBinding(binding, builder_->Normalize(new_value));
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple_node) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      ExprMutator::VisitBinding_(binding, tuple_node);
      return;
    }
    ffi::ObjectPtr<TupleNode> new_tuple = ffi::make_object<TupleNode>(*tuple_node);
    new_tuple->fields = RemapArgs(tuple_node->fields);
    new_tuple->ty = Type::Missing();
    Expr new_value = builder_->Normalize(Tuple(new_tuple));
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      NType to = NTypeFrom(binding->var);
      new_value = RewriteExpr(new_value, to);
    }
    ReEmitBinding(binding, builder_->Normalize(new_value));
  }

  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item_node) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      ExprMutator::VisitBinding_(binding, tuple_get_item_node);
      return;
    }
    ffi::ObjectPtr<TupleGetItemNode> new_tuple_get_item =
        ffi::make_object<TupleGetItemNode>(*tuple_get_item_node);
    new_tuple_get_item->tuple = RemapArgs({tuple_get_item_node->tuple})[0];
    new_tuple_get_item->ty = Type::Missing();
    Expr new_value = TupleGetItem(new_tuple_get_item);
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      NType to = NTypeFrom(binding->var);
      new_value = RewriteExpr(new_value, to);
    }
    ReEmitBinding(binding, builder_->Normalize(new_value));
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) {
    builder_->BeginDataflowBlock();
    for (auto param : params_) {
      CastIfDstOnly(param);
    }
    for (auto binding : block->bindings) {
      this->VisitBinding(binding);
    }
    for (auto param : params_) {
      auto it = var_remap_.find(param);
      if (it != var_remap_.end()) {
        var_remap_.erase(it);
      }
    }
    return builder_->EndBlock();
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    params_ = op->params;
    return ExprMutator::VisitExpr_(op);
  }

  const VarDTypeMap* only_dst_map_;
  CustomMixedPrecisionConfig config_;
  ffi::Array<Var> params_;
  std::unordered_set<std::string> dst_input_names_;
  const Op& wrap_param_op = Op::Get("relax.wrap_param");
};

Expr ToMixedPrecisionCustom(const Function& f, const CustomMixedPrecisionConfig& config,
                            ffi::Optional<ffi::Array<ffi::String>> dst_input_names) {
  VarDTypeMap only_dst_map = CustomDTypeDecisionCollector::Collect(f, config);
  std::unordered_set<std::string> dst_input_names_set;
  if (dst_input_names) {
    dst_input_names_set.insert(dst_input_names.value().begin(), dst_input_names.value().end());
  }
  ToMixedPrecisionCustomRewriter mutator(&only_dst_map, config, dst_input_names_set);
  return mutator(f);
}

}  // namespace relax
}  // namespace tvm
