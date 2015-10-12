#include "ExpressionRange.h"
#include "ExtractFromTime.h"
#include "GroupByAndAggregate.h"
#include "Execute.h"

#include <cfenv>

#define DEF_OPERATOR(fname, op)                                                                                    \
  ExpressionRange fname(const ExpressionRange& other) const {                                                      \
    return (type == ExpressionRangeType::Integer && other.type == ExpressionRangeType::Integer)                    \
               ? binOp<int64_t>(other,                                                                             \
                                [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) op y); }) \
               : binOp<double>(other, [](const double x, const double y) {                                         \
      std::feclearexcept(FE_OVERFLOW);                                                                             \
      std::feclearexcept(FE_UNDERFLOW);                                                                            \
      auto result = x op y;                                                                                        \
      if (std::fetestexcept(FE_OVERFLOW) || std::fetestexcept(FE_UNDERFLOW)) {                                     \
        throw std::runtime_error("overflow / underflow");                                                          \
      }                                                                                                            \
      return result;                                                                                               \
                 });                                                                                               \
  }

DEF_OPERATOR(ExpressionRange::operator+, +)
DEF_OPERATOR(ExpressionRange::operator-, -)
DEF_OPERATOR(ExpressionRange::operator*, *)

ExpressionRange ExpressionRange::operator/(const ExpressionRange& other) const {
  if (type != ExpressionRangeType::Integer || other.type != ExpressionRangeType::Integer) {
    return {ExpressionRangeType::Invalid, false, {0}, {0}};
  }
  if (other.int_min * other.int_max <= 0) {
    // if the other interval contains 0, the rule is more complicated;
    // punt for now, we can revisit by splitting the other interval and
    // taking the convex hull of the resulting two intervals
    return {ExpressionRangeType::Invalid, false, {0}, {0}};
  }
  return binOp<int64_t>(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) / y); });
}

ExpressionRange ExpressionRange::operator||(const ExpressionRange& other) const {
  if (type != other.type) {
    return {ExpressionRangeType::Invalid, false, {0}, {0}};
  }
  ExpressionRange result;
  switch (type) {
    case ExpressionRangeType::Invalid:
      return {ExpressionRangeType::Invalid, false, {0}, {0}};
    case ExpressionRangeType::Integer: {
      result.type = ExpressionRangeType::Integer;
      result.has_nulls = has_nulls || other.has_nulls;
      result.int_min = std::min(int_min, other.int_min);
      result.int_max = std::max(int_max, other.int_max);
      break;
    }
    case ExpressionRangeType::FloatingPoint: {
      result.type = ExpressionRangeType::FloatingPoint;
      result.has_nulls = has_nulls || other.has_nulls;
      result.fp_min = std::min(fp_min, other.fp_min);
      result.fp_max = std::max(fp_max, other.fp_max);
      break;
    }
    default:
      CHECK(false);
  }
  return result;
}

ExpressionRange getExpressionRange(const Analyzer::BinOper* expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::Constant* expr);

ExpressionRange getExpressionRange(const Analyzer::ColumnVar* col_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos);

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr);

ExpressionRange getExpressionRange(const Analyzer::CaseExpr* case_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::UOper* u_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::ExtractExpr* extract_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::Expr* expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor* executor) {
  auto bin_oper_expr = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper_expr) {
    return getExpressionRange(bin_oper_expr, query_infos, executor);
  }
  auto constant_expr = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant_expr) {
    return getExpressionRange(constant_expr);
  }
  auto column_var_expr = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (column_var_expr) {
    return getExpressionRange(column_var_expr, query_infos);
  }
  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    return getExpressionRange(like_expr);
  }
  auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_expr) {
    return getExpressionRange(case_expr, query_infos, executor);
  }
  auto u_expr = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_expr) {
    return getExpressionRange(u_expr, query_infos, executor);
  }
  auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
  if (extract_expr) {
    return getExpressionRange(extract_expr, query_infos, executor);
  }
  return {ExpressionRangeType::Invalid, false, {0}, {0}};
}

ExpressionRange getExpressionRange(const Analyzer::BinOper* expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor* executor) {
  const auto& lhs = getExpressionRange(expr->get_left_operand(), query_infos, executor);
  const auto& rhs = getExpressionRange(expr->get_right_operand(), query_infos, executor);
  switch (expr->get_optype()) {
    case kPLUS:
      return lhs + rhs;
    case kMINUS:
      return lhs - rhs;
    case kMULTIPLY:
      return lhs * rhs;
    case kDIVIDE:
      return lhs / rhs;
    default:
      break;
  }
  return {ExpressionRangeType::Invalid, false, {0}, {0}};
}

ExpressionRange getExpressionRange(const Analyzer::Constant* constant_expr) {
  if (constant_expr->get_is_null()) {
    return {ExpressionRangeType::Invalid, false, {0}, {0}};
  }
  const auto constant_type = constant_expr->get_type_info().get_type();
  const auto datum = constant_expr->get_constval();
  switch (constant_type) {
    case kSMALLINT: {
      const auto v = datum.smallintval;
      return {ExpressionRangeType::Integer, false, {v}, {v}};
    }
    case kINT: {
      const auto v = datum.intval;
      return {ExpressionRangeType::Integer, false, {v}, {v}};
    }
    case kBIGINT: {
      const auto v = datum.bigintval;
      return {ExpressionRangeType::Integer, false, {v}, {v}};
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      const auto v = datum.timeval;
      return {ExpressionRangeType::Integer, false, {v}, {v}};
    }
    case kFLOAT:
    case kDOUBLE: {
      const double v = constant_type == kDOUBLE ? datum.doubleval : datum.floatval;
      ExpressionRange result;
      result.type = ExpressionRangeType::FloatingPoint;
      result.has_nulls = false;
      result.fp_min = v;
      result.fp_max = v;
      return result;
    }
    case kNUMERIC:
    case kDECIMAL: {
      ExpressionRange result;
      result.type = ExpressionRangeType::FloatingPoint;
      result.has_nulls = false;
      result.fp_min = result.fp_max =
          static_cast<double>(datum.bigintval) / exp_to_scale(constant_expr->get_type_info().get_scale());
      return result;
    }
    default:
      break;
  }
  return {ExpressionRangeType::Invalid, false, {0}, {0}};
}

#define FIND_STAT_FRAG(stat_name)                                                                            \
  const auto stat_name##_frag =                                                                              \
      std::stat_name##_element(fragments.begin(),                                                            \
                               fragments.end(),                                                              \
                               [&has_nulls, col_id, col_ti](const Fragmenter_Namespace::FragmentInfo& lhs,   \
                                                            const Fragmenter_Namespace::FragmentInfo& rhs) { \
    auto lhs_meta_it = lhs.chunkMetadataMap.find(col_id);                                                    \
    if (lhs_meta_it == lhs.chunkMetadataMap.end()) {                                                         \
      return false;                                                                                          \
    }                                                                                                        \
    auto rhs_meta_it = rhs.chunkMetadataMap.find(col_id);                                                    \
    CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());                                                        \
    if (lhs_meta_it->second.chunkStats.has_nulls || rhs_meta_it->second.chunkStats.has_nulls) {              \
      has_nulls = true;                                                                                      \
    }                                                                                                        \
    if (col_ti.is_fp()) {                                                                                    \
      return extract_##stat_name##_stat_double(lhs_meta_it->second.chunkStats, col_ti) <                     \
             extract_##stat_name##_stat_double(rhs_meta_it->second.chunkStats, col_ti);                      \
    }                                                                                                        \
    return extract_##stat_name##_stat(lhs_meta_it->second.chunkStats, col_ti) <                              \
           extract_##stat_name##_stat(rhs_meta_it->second.chunkStats, col_ti);                               \
      });                                                                                                    \
  if (stat_name##_frag == fragments.end()) {                                                                 \
    return {ExpressionRangeType::Invalid, false, {0}, {0}};                                                  \
  }

namespace {

inline double extract_min_stat_double(const ChunkStats& stats, const SQLTypeInfo& col_ti) {
  if (stats.has_nulls) {  // clobber the additional information for now
    return NULL_DOUBLE;
  }
  return col_ti.get_type() == kDOUBLE ? stats.min.doubleval : stats.min.floatval;
}

inline double extract_max_stat_double(const ChunkStats& stats, const SQLTypeInfo& col_ti) {
  return col_ti.get_type() == kDOUBLE ? stats.max.doubleval : stats.max.floatval;
}

}  // namespace

ExpressionRange getExpressionRange(const Analyzer::ColumnVar* col_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos) {
  int col_id = col_expr->get_column_id();
  const auto& col_ti =
      col_expr->get_type_info().is_array() ? col_expr->get_type_info().get_elem_type() : col_expr->get_type_info();
  switch (col_ti.get_type()) {
    case kTEXT:
    case kCHAR:
    case kVARCHAR:
      CHECK_EQ(kENCODING_DICT, col_ti.get_compression());
    case kBOOLEAN:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kDECIMAL:
    case kNUMERIC:
    case kDATE:
    case kTIMESTAMP:
    case kTIME:
    case kFLOAT:
    case kDOUBLE: {
      bool has_nulls{false};
      const auto& fragments = query_infos[col_expr->get_rte_idx()].fragments;
      FIND_STAT_FRAG(min);
      FIND_STAT_FRAG(max);
      const auto min_it = min_frag->chunkMetadataMap.find(col_id);
      if (min_it == min_frag->chunkMetadataMap.end()) {
        return {ExpressionRangeType::Invalid, false, {0}, {0}};
      }
      const auto max_it = max_frag->chunkMetadataMap.find(col_id);
      CHECK(max_it != max_frag->chunkMetadataMap.end());
      if (col_ti.is_fp()) {
        ExpressionRange result;
        result.type = ExpressionRangeType::FloatingPoint;
        result.has_nulls = has_nulls;
        result.fp_min = extract_min_stat_double(min_it->second.chunkStats, col_ti);
        result.fp_max = extract_max_stat_double(max_it->second.chunkStats, col_ti);
        return result;
      }
      for (const auto& fragment : fragments) {
        const auto it = fragment.chunkMetadataMap.find(col_id);
        if (it != fragment.chunkMetadataMap.end()) {
          if (it->second.chunkStats.has_nulls) {
            has_nulls = true;
          }
        }
      }
      const auto min_val = extract_min_stat(min_it->second.chunkStats, col_ti);
      const auto max_val = extract_max_stat(max_it->second.chunkStats, col_ti);
      CHECK_GE(max_val, min_val);
      return {ExpressionRangeType::Integer, has_nulls, {min_val}, {max_val}};
    }
    default:
      break;
  }
  return {ExpressionRangeType::Invalid, false, {0}, {0}};
}

#undef FIND_STAT_FRAG

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr) {
  const auto& ti = like_expr->get_type_info();
  CHECK(ti.is_boolean());
  const auto& arg_ti = like_expr->get_arg()->get_type_info();
  return {ExpressionRangeType::Integer, false, {arg_ti.get_notnull() ? 0 : inline_int_null_val(ti)}, {1}};
}

ExpressionRange getExpressionRange(const Analyzer::CaseExpr* case_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor* executor) {
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  ExpressionRange expr_range{ExpressionRangeType::Invalid, false, {0}, {0}};
  const auto& case_ti = case_expr->get_type_info();
  for (const auto& expr_pair : expr_pair_list) {
    CHECK_EQ(expr_pair.first->get_type_info().get_type(), kBOOLEAN);
    CHECK(expr_pair.second->get_type_info() == case_ti);
    const auto crt_range = getExpressionRange(expr_pair.second.get(), query_infos, executor);
    if (crt_range.type == ExpressionRangeType::Invalid) {
      return {ExpressionRangeType::Invalid, false, {0}, {0}};
    }
    expr_range = (expr_range.type != ExpressionRangeType::Invalid) ? expr_range || crt_range : crt_range;
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  return expr_range || getExpressionRange(else_expr, query_infos, executor);
}

ExpressionRange getExpressionRange(const Analyzer::UOper* u_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor* executor) {
  if (u_expr->get_optype() == kUNNEST) {
    return getExpressionRange(u_expr->get_operand(), query_infos, executor);
  }
  if (u_expr->get_optype() != kCAST) {
    return {ExpressionRangeType::Invalid, false, {0}, {0}};
  }
  const auto& ti = u_expr->get_type_info();
  if (ti.is_string() && ti.get_compression() == kENCODING_DICT) {
    const auto sd = executor->getStringDictionary(ti.get_comp_param(), nullptr);
    CHECK(sd);
    const auto const_operand = dynamic_cast<const Analyzer::Constant*>(u_expr->get_operand());
    CHECK(const_operand);
    CHECK(const_operand->get_constval().stringval);
    ExpressionRange expr_range;
    expr_range.type = ExpressionRangeType::Integer;
    expr_range.has_nulls = false;
    expr_range.int_min = expr_range.int_max = sd->get(*const_operand->get_constval().stringval);
    return expr_range;
  }
  const auto arg_range = getExpressionRange(u_expr->get_operand(), query_infos, executor);
  switch (arg_range.type) {
    case ExpressionRangeType::FloatingPoint: {
      const auto& arg_ti = u_expr->get_operand()->get_type_info();
      if (ti.is_fp() && ti.get_size() >= arg_ti.get_size()) {
        return arg_range;
      }
      if (ti.is_integer()) {
        ExpressionRange result;
        result.type = ExpressionRangeType::Integer;
        result.has_nulls = arg_range.has_nulls;
        result.int_min = arg_range.fp_min;
        result.int_max = arg_range.fp_max;
        return result;
      }
      break;
    }
    case ExpressionRangeType::Integer: {
      if (ti.is_integer() || ti.is_time()) {
        return arg_range;
      }
      if (ti.is_decimal()) {
        ExpressionRange result;
        result.type = ExpressionRangeType::FloatingPoint;
        result.has_nulls = arg_range.has_nulls;
        result.fp_min = arg_range.int_min;
        result.fp_max = arg_range.int_max;
        return result;
      }
      if (ti.get_type() == kDOUBLE) {
        ExpressionRange result;
        result.type = ExpressionRangeType::Integer;
        result.has_nulls = arg_range.has_nulls;
        result.fp_min = arg_range.int_min;
        result.fp_max = arg_range.int_max;
        return result;
      }
      break;
    }
    case ExpressionRangeType::Invalid:
      break;
    default:
      CHECK(false);
  }
  return {ExpressionRangeType::Invalid, false, {0}, {0}};
}

ExpressionRange getExpressionRange(const Analyzer::ExtractExpr* extract_expr,
                                   const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                   const Executor* executor) {
  const int32_t extract_field{extract_expr->get_field()};
  ExpressionRange result;
  result.type = ExpressionRangeType::Integer;
  const auto arg_range = getExpressionRange(extract_expr->get_from_expr(), query_infos, executor);
  result.has_nulls = arg_range.type == ExpressionRangeType::Invalid || arg_range.has_nulls;
  switch (extract_field) {
    case kYEAR: {
      auto year_range = arg_range;
      if (year_range.type == ExpressionRangeType::Invalid) {
        return {ExpressionRangeType::Invalid, false, {0}, {0}};
      }
      CHECK(year_range.type == ExpressionRangeType::Integer);
      year_range.int_min = ExtractFromTime(kYEAR, year_range.int_min);
      year_range.int_max = ExtractFromTime(kYEAR, year_range.int_max);
      return year_range;
    }
    case kEPOCH:
      return arg_range;
    case kMONTH:
      result.int_min = 1;
      result.int_max = 12;
      break;
    case kDAY:
      result.int_min = 1;
      result.int_max = 31;
      break;
    case kHOUR:
      result.int_min = 0;
      result.int_max = 23;
      break;
    case kMINUTE:
      result.int_min = 0;
      result.int_max = 59;
      break;
    case kSECOND:
      result.int_min = 0;
      result.int_max = 60;
      break;
    case kDOW:
      result.int_min = 0;
      result.int_max = 6;
      break;
    case kDOY:
      result.int_min = 1;
      result.int_max = 366;
      break;
    default:
      CHECK(false);
  }
  return result;
}
