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
 * 
 * This is an extended version of to_mixed_precision.cc that supports
 * custom datatypes like Posit. It allows mixed precision between:
 * - Posit32 -> Posit16 with Posit32 accumulation
 * - Posit16 -> Posit8 with Posit16 accumulation
 * - Any custom dtype pairs
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <unordered_set>

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
 * 
 * Replaces the hardcoded fp32/fp16 with configurable src/dst datatypes.
 */
struct CustomMixedPrecisionConfig {
  DataType src_dtype;   // Source dtype (e.g., posit32, float32)
  DataType dst_dtype;   // Destination dtype (e.g., posit16, float16)
  DataType acc_dtype;   // Accumulator dtype (e.g., posit32, float32)

  static CustomMixedPrecisionConfig FromStrings(
      const std::string& src, const std::string& dst, const std::string& acc) {
    return CustomMixedPrecisionConfig{
      DataType(ffi::StringToDLDataType(src)),
      DataType(ffi::StringToDLDataType(dst)),
      DataType(ffi::StringToDLDataType(acc))
    };
  }
};

/*!
 * \brief Parse dtype information from string
 * \return tuple of (family, bits, es) where family is 0=float, 1=posit, 2=other
 */
static bool ParsePositiveInt(const std::string& s, int* value) {
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  try {
    *value = std::stoi(s);
    return *value > 0;
  } catch (...) {
    return false;
  }
}

static std::tuple<int, int, int> ParseDTypeInfo(const std::string& dtype_str) {
  // Float family: floatX, e.g. float8/16/32/64.
  if (dtype_str.rfind("float", 0) == 0) {
    int bits = 0;
    if (ParsePositiveInt(dtype_str.substr(5), &bits)) {
      return std::make_tuple(0, bits, -1);
    }
  }

  // Posit family: custom[posites<es>]<bits>.
  if (dtype_str.rfind("custom[posites", 0) == 0) {
    size_t prefix_len = std::string("custom[posites").size();
    size_t bracket_end = dtype_str.find(']', prefix_len);
    if (bracket_end != std::string::npos && bracket_end + 1 < dtype_str.size()) {
      int es = 0;
      int bits = 0;
      std::string es_str = dtype_str.substr(prefix_len, bracket_end - prefix_len);
      std::string bits_str = dtype_str.substr(bracket_end + 1);
      if (ParsePositiveInt(es_str, &es) && ParsePositiveInt(bits_str, &bits)) {
        return std::make_tuple(1, bits, es);
      }
    }
  }

  return std::make_tuple(2, 0, -1);
}

/*!
 * \brief Validate dtype combination for mixed precision
 * \param src Source dtype string
 * \param dst Destination dtype string
 * \param acc Accumulator dtype string
 * \throws Error if invalid combination
 */
static void ValidateMixedPrecisionDTypes(const std::string& src, 
                                          const std::string& dst, 
                                          const std::string& acc) {
  auto src_info = ParseDTypeInfo(src);
  auto dst_info = ParseDTypeInfo(dst);
  auto acc_info = ParseDTypeInfo(acc);
  
  int src_family = std::get<0>(src_info);
  int dst_family = std::get<0>(dst_info);
  int acc_family = std::get<0>(acc_info);
  
  int src_bits = std::get<1>(src_info);
  int dst_bits = std::get<1>(dst_info);
  int acc_bits = std::get<1>(acc_info);
  
  int src_es = std::get<2>(src_info);
  int dst_es = std::get<2>(dst_info);
  int acc_es = std::get<2>(acc_info);
  
  // Rule 1: Mixed precision is only supported for posit dtypes.
  if (!(src_family == 1 && dst_family == 1 && acc_family == 1)) {
    LOG(FATAL) << "Invalid dtype combination for mixed precision: "
               << "Mixed precision only supports posit dtypes with the same configuration. "
               << "Got: src=" << src << " (family=" << src_family << "), "
               << "dst=" << dst << " (family=" << dst_family << "), "
               << "acc=" << acc << " (family=" << acc_family << "). "
               << "For simple dtype conversion (without mixed precision restrictions), "
               << "use ChangeDatatype with use_mixed_precision=False in Python.";
  }

  // Rule 2: For posit types involved, es must be the same.
  int posit_es = -1;
  auto check_posit_es = [&](int family, int es, const char* dtype_name, const std::string& dtype) {
    if (family != 1) {
      return;
    }
    if (posit_es == -1) {
      posit_es = es;
      return;
    }
    if (posit_es != es) {
      LOG(FATAL) << "Invalid dtype combination for mixed precision: "
                 << "All posit dtypes must use the same 'es' parameter. "
                 << "Mismatch at " << dtype_name << "=" << dtype << " (es=" << es
                 << "), expected es=" << posit_es << ".";
    }
  };
  check_posit_es(src_family, src_es, "src", src);
  check_posit_es(dst_family, dst_es, "dst", dst);
  check_posit_es(acc_family, acc_es, "acc", acc);

  if (acc_bits > 0 && dst_bits > 0 && acc_bits < dst_bits) {
    LOG(FATAL) << "Invalid dtype combination for mixed precision: "
               << "Accumulation dtype should have equal or higher precision than dst. "
               << "Got: dst=" << dst << " (" << dst_bits << " bits), "
               << "acc=" << acc << " (" << acc_bits << " bits).";
  }

  if (src_bits > 0 && dst_bits > 0 && dst_bits > src_bits) {
    LOG(FATAL) << "Invalid dtype combination for mixed precision: "
               << "Destination dtype should not have higher precision than source. "
               << "Got: src=" << src << " (" << src_bits << " bits), "
               << "dst=" << dst << " (" << dst_bits << " bits).";
  }
}

/*!
 * \brief DType decision collector with custom dtype support.
 * 
 * Replaces hardcoded fp16_/fp32_ with configurable src/dst dtypes.
 */
class CustomDTypeDecisionCollector : public ExprVisitor {
 public:
  explicit CustomDTypeDecisionCollector(const CustomMixedPrecisionConfig& config)
      : config_(config),
        unknown_(DataType(DataType::TypeCode::kFloat, 0, 1)),
        src_dtype_(config.src_dtype),
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
      // Don't automatically infer and store - just return unknown
      // This avoids storing stale type info from struct_info in multi-stage conversions
      return NTypeFrom(var, unknown_);
    }
    return it->second;
  }

  // Create NType with void/unknown dtype (don't infer from struct_info)
  NType NTypeUnknown(const StructInfo& sinfo) {
    auto fmapleaf = [&](const StructInfo& sinfo) -> NType {
      // Always return empty string (unknown/void dtype)
      return NType(ffi::String(""));
    };
    return MapToNestedMsg<ffi::String>(sinfo, fmapleaf);
  }

  NType GetOrInferDType(const Var& var) {
    auto it = dtype_map_.find(var);
    if (it == dtype_map_.end()) {
      NType inferred = NTypeFrom(var, unknown_);
      dtype_map_[var] = inferred;
      return inferred;
    }
    return it->second;
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
    ICHECK(args.size() == to.size()) << "Invalid target dtypes";
    for (size_t i = 0; i < args.size(); ++i) {
      auto fvisitleaf = [&](const Expr& expr, NType to) {
        if (const auto* var = expr.as<VarNode>()) {
          UpdateVarDTypeMap(ffi::GetRef<Var>(var), to);
        } else if (expr->IsInstance<ConstantNode>()) {
          // Constant can be casted anyway
          return;
        } else {
          LOG(FATAL) << "Unsupported argument type: " << expr->GetTypeKey();
        }
      };
      DecomposeNestedMsg(args[i], to[i], fvisitleaf);
    }
  }

  void RequireArgsToType(ffi::Array<Expr> args, DataType to) {
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

  void VisitVars_(const VarNode* op) {
    Var var = ffi::GetRef<Var>(op);
    if (IsNestedTensor(var)) {
      // Require the var to be src_dtype (original dtype)
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
    if (policy == kAlways) {
      // Require inputs to be dst_dtype (e.g., posit16)
      RequireArgsToType(call_node->args, dst_dtype_);
    } else if (policy == kFollow || policy == kNever) {
      // Require inputs to be src_dtype (e.g., posit32)
      RequireArgsToType(call_node->args, src_dtype_);
    } else {
      LOG(FATAL) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple_node) final {
    NType lhs_type = GetDType(binding->var);
    RequireArgsToType(tuple_node->fields, lhs_type.NestedArray());
  }

  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item_node) final {
    NType lhs_type = GetDType(binding->var);
    
    // Handle TupleGetItem: construct the complete Tuple type
    const TupleStructInfoNode* sinfo =
        tuple_get_item_node->tuple->struct_info_.as<TupleStructInfoNode>();
    ICHECK(sinfo != nullptr) << "TupleGetItemNode must have TupleStructInfo";
    
    // Check if the tuple is a variable that we'll visit later (in backward traversal)
    const VarNode* tuple_var = tuple_get_item_node->tuple.as<VarNode>();
    bool tuple_has_type = false;
    if (tuple_var != nullptr) {
      Var tuple_v = ffi::GetRef<Var>(tuple_var);
      tuple_has_type = (dtype_map_.find(tuple_v) != dtype_map_.end());
    }
    
    std::vector<NType> require_rhs;
    for (size_t i = 0; i < sinfo->fields.size(); ++i) {
      if (i == static_cast<size_t>(tuple_get_item_node->index)) {
        // The selected element should match lhs
        require_rhs.push_back(lhs_type);
      } else {
        // For non-selected elements:
        // - If tuple var already has type info, use existing type
        // - Otherwise use pure unknown (not inferred from struct_info which may be outdated)
        if (tuple_has_type && tuple_var != nullptr) {
          Var tuple_v = ffi::GetRef<Var>(tuple_var);
          NType existing_type = dtype_map_[tuple_v];
          if (!existing_type.IsLeaf()) {
            ffi::Array<NType> existing_arr = existing_type.NestedArray();
            if (i < existing_arr.size()) {
              require_rhs.push_back(existing_arr[i]);
              continue;
            }
          }
        }
        // Use pure unknown type (empty string) to avoid inferring from outdated struct_info
        require_rhs.push_back(NTypeUnknown(sinfo->fields[i]));
      }
    }
    RequireArgsToType({tuple_get_item_node->tuple}, {NType(require_rhs)});
  }

  // Override to visit in backward order
  void VisitExpr_(const SeqExprNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->body);
    for (auto it = op->blocks.rbegin(); it != op->blocks.rend(); it++) {
      this->VisitBindingBlock(*it);
    }
    if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
      this->VisitExprDepStructInfoField(ffi::GetRef<StructInfo>(sinfo));
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
    if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
      this->VisitExprDepStructInfoField(ffi::GetRef<StructInfo>(sinfo));
    }
  }

  CustomMixedPrecisionConfig config_;
  DataType unknown_;
  DataType src_dtype_;
  DataType dst_dtype_;
  VarDTypeMap dtype_map_;
};

// Forward declaration of ToMixedPrecisionCustom
Expr ToMixedPrecisionCustom(const Function& f, const CustomMixedPrecisionConfig& config,
                            ffi::Optional<ffi::Array<ffi::String>> dst_input_names);

namespace transform {

/*!
 * \brief Create mixed precision pass with custom datatype support.
 * 
 * \param src_dtype Source datatype string (e.g., "float32", "custom[posites2]32")
 * \param dst_dtype Destination datatype string (e.g., "float16", "custom[posites2]16")
 * \param acc_dtype Accumulator datatype string (e.g., "float32", "custom[posites2]32")
 * \param dst_input_names Optional list of parameter names to convert to dst_dtype
 * \return The transform pass
 */
Pass ToMixedPrecisionCustom(const std::string& src_dtype,
                            const std::string& dst_dtype,
                            const std::string& acc_dtype,
                            ffi::Optional<ffi::Array<ffi::String>> dst_input_names) {
  // Validate dtype combination before creating the pass
  ValidateMixedPrecisionDTypes(src_dtype, dst_dtype, acc_dtype);
  
  auto config = CustomMixedPrecisionConfig::FromStrings(src_dtype, dst_dtype, acc_dtype);
  
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(ToMixedPrecisionCustom(f, config, dst_input_names));
  };
  return CreateFunctionPass(pass_func, 0, "ToMixedPrecisionCustom", {});
}

/*!
 * \brief Convenience function for Posit32 -> Posit16 mixed precision.
 */
Pass ToMixedPrecisionPosit32ToPosit16(int es,
                                       ffi::Optional<ffi::Array<ffi::String>> dst_input_names) {
  std::string posit32_name = "custom[posites" + std::to_string(es) + "]32";
  std::string posit16_name = "custom[posites" + std::to_string(es) + "]16";
  return ToMixedPrecisionCustom(posit32_name, posit16_name, posit32_name, dst_input_names);
}

/*!
 * \brief Convenience function for Posit16 -> Posit8 mixed precision.
 */
Pass ToMixedPrecisionPosit16ToPosit8(int es,
                                      ffi::Optional<ffi::Array<ffi::String>> dst_input_names) {
  std::string posit16_name = "custom[posites" + std::to_string(es) + "]16";
  std::string posit8_name = "custom[posites" + std::to_string(es) + "]8";
  return ToMixedPrecisionCustom(posit16_name, posit8_name, posit16_name, dst_input_names);
}

/*!
 * \brief Generic convenience function for arbitrary posit mixed precision.
 */
Pass ToMixedPrecisionPosit(int es,
                           int src_bits,
                           int dst_bits,
                           int acc_bits,
                           ffi::Optional<ffi::Array<ffi::String>> dst_input_names) {
  ICHECK_GT(es, 0) << "es must be > 0";
  ICHECK_GT(src_bits, 0) << "src_bits must be > 0";
  ICHECK_GT(dst_bits, 0) << "dst_bits must be > 0";
  ICHECK_GT(acc_bits, 0) << "acc_bits must be > 0";

  auto build_posit = [es](int bits) {
    return "custom[posites" + std::to_string(es) + "]" + std::to_string(bits);
  };

  std::string src_name = build_posit(src_bits);
  std::string dst_name = build_posit(dst_bits);
  std::string acc_name = build_posit(acc_bits);
  return ToMixedPrecisionCustom(src_name, dst_name, acc_name, dst_input_names);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
    .def("relax.transform.ToMixedPrecisionCustom", ToMixedPrecisionCustom)
    .def("relax.transform.ToMixedPrecisionPosit32ToPosit16", ToMixedPrecisionPosit32ToPosit16)
    .def("relax.transform.ToMixedPrecisionPosit16ToPosit8", ToMixedPrecisionPosit16ToPosit8)
    .def("relax.transform.ToMixedPrecisionPosit", ToMixedPrecisionPosit);
}

}  // namespace transform

/*!
 * \brief ToMixedPrecisionCustomRewriter with configurable datatypes.
 * 
 * Replaces hardcoded fp16_/fp32_ with config src/dst/acc dtypes.
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
    auto it = var_remap_.find(var->vid);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      if (dst_input_names_.count(var->name_hint())) {
        auto sinfo = GetStructInfo(var);
        if (auto tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
          VDevice vdev = VDevice();
          if (tensor_sinfo->vdevice.defined()) {
            vdev = tensor_sinfo->vdevice.value();
          }
          TensorStructInfo dst_sinfo(tensor_sinfo->shape.value(), config_.dst_dtype, vdev,
                                     tensor_sinfo->span);
          Var dst_var(var->vid, dst_sinfo, var->span);
          var_remap_[var->vid] = dst_var;
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
      const auto* tensor = GetStructInfoAs<TensorStructInfoNode>(expr);
      ICHECK(tensor != nullptr) << "Only support rewriting tensor expr";
      
      if (NTypeEqual()(to[0], NTypeFrom(expr))) return expr;
      
      // Only rewrite if dtype is src or dst (not other types like int32)
      if (tensor->dtype != config_.src_dtype && 
          tensor->dtype != config_.dst_dtype &&
          tensor->dtype != config_.acc_dtype) {
        return expr;
      }
      
      return astype(expr, DataType(ffi::StringToDLDataType(to[0].LeafValue())));
    };
    return TransformTupleLeaf<ffi::String>(expr, std::array<NType, 1>({to}), fvisitleaf);
  }

  ffi::Array<Expr> RewriteArgs(const ffi::Array<Expr>& args, DataType to) {
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
    auto is_dst = [this](StructInfo sinfo) {
      if (auto tensor_sinfo = sinfo.as<TensorStructInfoNode>();
          tensor_sinfo && tensor_sinfo->dtype == config_.dst_dtype) {
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
      auto sinfo = GetStructInfo(arg);
      auto constant = arg.as<ConstantNode>();
      auto tuple = arg.as<TupleNode>();

      if (!IsNestedTensor(arg) || is_dst(sinfo) || 
          (constant && is_in_dst_range(constant)) ||
          (tuple && AllDstCastable(tuple->fields))) {
        continue;
      } else {
        return false;
      }
    }

    return true;
  }

  void CastIfDstOnly(const Var& var) {
    ICHECK(builder_->CurrentBlockIsDataFlow());
    Var cur_var = GetRemapped(var);
    
    auto it = only_dst_map_->find(var);
    if (it == only_dst_map_->end()) return;
    
    // Get the to dtype
    auto dst_dtype_str = ffi::DLDataTypeToString(config_.dst_dtype);
    auto fcombine = [&dst_dtype_str](const ffi::String& from, const ffi::String& required) -> ffi::String {
      return required == dst_dtype_str ? required : from;
    };
    NType from = NTypeFrom(cur_var);
    NType to = CombineNestedMsg<ffi::String>(from, it->second, fcombine);
    Expr rewrite = RewriteExpr(cur_var, to);
    
    if (!rewrite.same_as(cur_var)) {
      var_remap_[var->vid] = builder_->Emit(rewrite);
    }
  }

  Expr VisitVar_(const Var& var) {
    auto it = var_remap_.find(var->vid);
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
    ICHECK(op_node != nullptr);
    Op op = ffi::GetRef<Op>(op_node);
    if (wrap_param_op.same_as(op)) {
      ReEmitBinding(binding, call_node->args[0]);
      return;
    }

    Call new_call = ffi::GetRef<Call>(call_node);
    new_call.CopyOnWrite()->args = RemapArgs(new_call->args);

    std::optional<DataType> opt_new_dtype = std::nullopt;

    if (policy == kAlways) {
      // Use dst_dtype for computation, acc_dtype for output
      opt_new_dtype = config_.dst_dtype;
      auto attr_map = Op::GetAttrMap<FInferMixedPrecision>("FInferMixedPrecision");
      ICHECK(attr_map.count(op));
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
      LOG(FATAL) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }

    Expr new_value = new_call;
    if (opt_new_dtype) {
      auto new_dtype = opt_new_dtype.value();
      new_call.CopyOnWrite()->args = RewriteArgs(new_call->args, new_dtype);
      new_call.CopyOnWrite()->struct_info_ = std::nullopt;

      new_value = builder_->Normalize(Call(new_call));

      if (!binding->var->IsInstance<DataflowVarNode>()) {
        new_value = RewriteExpr(new_value, NTypeFrom(binding->var));
      } else if (policy == kAlways && binding->var->IsInstance<DataflowVarNode>()) {
        new_value = RewriteExpr(new_value, NTypeFrom(new_value, new_dtype));
      }
    }

    ReEmitBinding(binding, builder_->Normalize(new_value));
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple_node) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      ExprMutator::VisitBinding_(binding, tuple_node);
      return;
    }
    ObjectPtr<TupleNode> new_tuple = ffi::make_object<TupleNode>(*tuple_node);
    new_tuple->fields = RemapArgs(tuple_node->fields);
    new_tuple->struct_info_ = std::nullopt;
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
    ObjectPtr<TupleGetItemNode> new_tuple_get_item =
        ffi::make_object<TupleGetItemNode>(*tuple_get_item_node);
    new_tuple_get_item->tuple = RemapArgs({tuple_get_item_node->tuple})[0];
    new_tuple_get_item->struct_info_ = std::nullopt;
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
      auto it = var_remap_.find(param->vid);
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
