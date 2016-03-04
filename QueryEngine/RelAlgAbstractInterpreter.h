#ifndef QUERYENGINE_RELALGABSTRACTINTERPRETER_H
#define QUERYENGINE_RELALGABSTRACTINTERPRETER_H

#include "../Catalog/Catalog.h"

#include <boost/variant.hpp>
#include <rapidjson/document.h>

#include <memory>

class Rex {
 public:
  virtual std::string toString() const = 0;
};

class RexScalar : public Rex {};

// For internal use of the abstract interpreter only. The result after abstract
// interpretation will not have any references to 'RexAbstractInput' objects.
class RexAbstractInput : public RexScalar {
 public:
  RexAbstractInput(const unsigned in_index) : in_index_(in_index) {}

  unsigned getIndex() const { return in_index_; }

  std::string toString() const override { return "(RexAbstractInput " + std::to_string(in_index_) + ")"; }

 private:
  unsigned in_index_;
};

class RexLiteral : public RexScalar {
 public:
  RexLiteral(const int64_t val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kDECIMAL, type);
  }

  RexLiteral(const double val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kDOUBLE, type);
  }

  RexLiteral(const std::string& val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kTEXT, type);
  }

  RexLiteral(const bool val,
             const SQLTypes type,
             const unsigned scale,
             const unsigned precision,
             const unsigned type_scale,
             const unsigned type_precision)
      : literal_(val),
        type_(type),
        scale_(scale),
        precision_(precision),
        type_scale_(type_scale),
        type_precision_(type_precision) {
    CHECK_EQ(kBOOLEAN, type);
  }

  RexLiteral() : literal_(nullptr), type_(kNULLT), scale_(0), precision_(0), type_scale_(0), type_precision_(0) {}

  template <class T>
  T getVal() const {
    const auto ptr = boost::get<T>(&literal_);
    CHECK(ptr);
    return *ptr;
  }

  SQLTypes getType() const { return type_; }

  unsigned getScale() const { return scale_; }

  unsigned getPrecision() const { return precision_; }

  unsigned getTypeScale() const { return type_scale_; }

  unsigned getTypePrecision() const { return type_precision_; }

  std::string toString() const override { return "(RexLiteral " + boost::lexical_cast<std::string>(literal_) + ")"; }

 private:
  const boost::variant<int64_t, double, std::string, bool, void*> literal_;
  const SQLTypes type_;
  const unsigned scale_;
  const unsigned precision_;
  const unsigned type_scale_;
  const unsigned type_precision_;
};

class RexOperator : public RexScalar {
 public:
  RexOperator(const SQLOps op, const std::vector<const RexScalar*> operands) : op_(op) {
    for (auto operand : operands) {
      operands_.emplace_back(operand);
    }
  }

  size_t size() const { return operands_.size(); }

  const RexScalar* getOperand(const size_t idx) const {
    CHECK(idx < operands_.size());
    return operands_[idx].get();
  }

  SQLOps getOperator() const { return op_; }

  std::string toString() const override {
    std::string result = "(RexOperator " + std::to_string(op_);
    for (const auto& operand : operands_) {
      result += " " + operand->toString();
    }
    return result + ")";
  };

 private:
  const SQLOps op_;
  std::vector<std::unique_ptr<const RexScalar>> operands_;
};

class RexCast : public RexOperator {
 public:
  RexCast(const RexScalar* operand, const SQLTypes target_type, const bool nullable)
      : RexOperator(kCAST, {operand}), target_type_(target_type), nullable_(nullable) {}

  SQLTypes getTargetType() const { return target_type_; }

  bool getNullable() const { return nullable_; }

  std::string toString() const override {
    return "(RexCast " + getOperand(0)->toString() + " to " + std::to_string(target_type_) + ")";
  }

 private:
  SQLTypes target_type_;
  const bool nullable_;
};

class RelAlgNode;

// The actual input node understood by the Executor.
// The in_index_ is relative to the output of node_.
class RexInput : public RexAbstractInput {
 public:
  RexInput(const RelAlgNode* node, const unsigned in_index) : RexAbstractInput(in_index), node_(node) {}

  const RelAlgNode* getSourceNode() const { return node_; }

  std::string toString() const override {
    return "(RexInput " + std::to_string(getIndex()) + " " + std::to_string(reinterpret_cast<const uint64_t>(node_)) +
           ")";
  }

 private:
  const RelAlgNode* node_;
};

class RexAgg : public Rex {
 public:
  RexAgg(const SQLAgg agg, const bool distinct, const SQLTypes type, const bool nullable, const ssize_t operand)
      : agg_(agg), distinct_(distinct), type_(type), nullable_(nullable), operand_(operand){};

  std::string toString() const override {
    return "(RexAgg " + std::to_string(agg_) + " " + std::to_string(distinct_) + " " +
           std::to_string(static_cast<int>(type_)) + " " + std::to_string(static_cast<int>(nullable_)) + " " +
           std::to_string(operand_) + ")";
  }

  SQLAgg getKind() const { return agg_; }

  bool isDistinct() const { return distinct_; }

  ssize_t getOperand() const { return operand_; }

 private:
  const SQLAgg agg_;
  const bool distinct_;
  const SQLTypes type_;
  const bool nullable_;
  const ssize_t operand_;
};

class RelAlgNode {
 public:
  RelAlgNode() : context_data_(nullptr) {}
  void setContextData(const void* context_data) const { context_data_ = context_data; }
  const size_t inputCount() const { return inputs_.size(); }
  const RelAlgNode* getInput(const size_t idx) const {
    CHECK(idx < inputs_.size());
    return inputs_[idx].get();
  }

  const RelAlgNode* getInputAndRelease(const size_t idx) {
    CHECK(idx < inputs_.size());
    return inputs_[idx].release();
  }

  const void addInput(const RelAlgNode* input) { inputs_.emplace_back(input); }

  bool replaceInput(const RelAlgNode* old_input, const RelAlgNode* input) {
    for (auto& input_ptr : inputs_) {
      if (input_ptr.get() == old_input) {
        input_ptr.reset(input);
        return true;
      }
    }
    return false;
  }

  virtual std::string toString() const = 0;

 protected:
  std::vector<std::unique_ptr<const RelAlgNode>> inputs_;
  mutable const void* context_data_;
};

class RelScan : public RelAlgNode {
 public:
  RelScan(const TableDescriptor* td, const std::vector<std::string>& field_names)
      : td_(td), field_names_(field_names) {}

  size_t size() const { return field_names_.size(); }

  const TableDescriptor* getTableDescriptor() const { return td_; }

  const std::vector<std::string>& getFieldNames() const { return field_names_; }

  std::string toString() const override {
    return "(RelScan<" + std::to_string(reinterpret_cast<uint64_t>(this)) + "> " +
           std::to_string(reinterpret_cast<uint64_t>(td_)) + ")";
  }

 private:
  const TableDescriptor* td_;
  const std::vector<std::string> field_names_;
};

class RelProject : public RelAlgNode {
 public:
  // Takes memory ownership of the expressions.
  RelProject(const std::vector<const RexScalar*>& exprs,
             const std::vector<std::string>& fields,
             const RelAlgNode* input)
      : fields_(fields) {
    CHECK_EQ(exprs.size(), fields.size());
    for (auto expr : exprs) {
      scalar_exprs_.emplace_back(expr);
    }
    inputs_.emplace_back(input);
  }

  void setExpressions(const std::vector<const RexScalar*>& exprs) {
    decltype(scalar_exprs_)().swap(scalar_exprs_);
    for (auto expr : exprs) {
      scalar_exprs_.emplace_back(expr);
    }
  }

  // True iff all the projected expressions are inputs. If true,
  // this node can be elided and merged into the previous node
  // since it's just a subset and / or permutation of its outputs.
  bool isSimple() const {
    for (const auto& expr : scalar_exprs_) {
      if (!dynamic_cast<const RexInput*>(expr.get())) {
        return false;
      }
    }
    return true;
  }

  size_t size() const { return scalar_exprs_.size(); }

  const RexScalar* getProjectAt(const size_t idx) const {
    CHECK(idx < scalar_exprs_.size());
    return scalar_exprs_[idx].get();
  }

  std::vector<const RexScalar*> getExpressionsAndRelease() {
    std::vector<const RexScalar*> result;
    for (auto& expr : scalar_exprs_) {
      result.push_back(expr.release());
    }
    return result;
  }

  const std::vector<std::string>& getFields() const { return fields_; }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  std::string toString() const override {
    std::string result = "(RelProject<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">";
    for (const auto& scalar_expr : scalar_exprs_) {
      result += " " + scalar_expr->toString();
    }
    return result + ")";
  }

 private:
  std::vector<std::unique_ptr<const RexScalar>> scalar_exprs_;
  const std::vector<std::string> fields_;
};

class RelAggregate : public RelAlgNode {
 public:
  // Takes ownership of the aggregate expressions.
  RelAggregate(const std::vector<size_t>& group_indices,
               const std::vector<const RexAgg*>& agg_exprs,
               const std::vector<std::string>& fields,
               const RelAlgNode* input)
      : group_indices_(group_indices), fields_(fields) {
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
    inputs_.emplace_back(input);
  }

  size_t size() const { return group_indices_.size() + agg_exprs_.size(); }

  const std::vector<size_t>& getGroupIndices() const { return group_indices_; }

  const std::vector<std::string>& getFields() const { return fields_; }

  std::vector<const RexAgg*> getAggregatesAndRelease() {
    std::vector<const RexAgg*> result;
    for (auto& agg_expr : agg_exprs_) {
      result.push_back(agg_expr.release());
    }
    return result;
  }

  std::string toString() const override {
    std::string result = "(RelAggregate<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(groups: [";
    for (const auto group_index : group_indices_) {
      result += " " + std::to_string(group_index);
    }
    result += " ] aggs: [";
    for (const auto& agg_expr : agg_exprs_) {
      result += " " + agg_expr->toString();
    }
    return result + " ])";
  }

 private:
  const std::vector<size_t> group_indices_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
};

enum class RelJoinType { INNER, LEFT };

class RelJoin : public RelAlgNode {
 public:
  RelJoin(const RelAlgNode* lhs, const RelAlgNode* rhs, const RexScalar* condition, const RelJoinType join_type)
      : condition_(condition), join_type_(join_type) {
    inputs_.emplace_back(lhs);
    inputs_.emplace_back(rhs);
  }

  std::string toString() const override {
    std::string result = "(RelJoin<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    result += condition_ ? condition_->toString() : "null";
    result += " " + std::to_string(static_cast<int>(join_type_));
    return result + ")";
  }

 private:
  const std::unique_ptr<const RexScalar> condition_;
  const RelJoinType join_type_;
};

class RelFilter : public RelAlgNode {
 public:
  RelFilter(const RexScalar* filter, const RelAlgNode* input) : filter_(filter) { inputs_.emplace_back(input); }

  const RexScalar* getCondition() const { return filter_.get(); }

  const RexScalar* getAndReleaseCondition() { return filter_.release(); }

  void setCondition(const RexScalar* condition) { filter_.reset(condition); }

  std::string toString() const override {
    std::string result = "(RelFilter<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    result += filter_ ? filter_->toString() : "null";
    return result + ")";
  }

 private:
  std::unique_ptr<const RexScalar> filter_;
};

// The 'RelCompound' node combines filter and on the fly aggregate computation.
// It's the result of combining a sequence of 'RelFilter' (optional), 'RelProject',
// 'RelAggregate' (optional) and a simple 'RelProject' (optional) into a single node
// which can be efficiently executed with no intermediate buffers.
class RelCompound : public RelAlgNode {
 public:
  // 'target_exprs_' are either scalar expressions owned by 'scalar_sources_'
  // or aggregate expressions owned by 'agg_exprs_', with the arguments
  // owned by 'scalar_sources_'.
  RelCompound(const RexScalar* filter_expr,
              const std::vector<const Rex*>& target_exprs,
              const std::vector<size_t>& group_indices,
              const std::vector<const RexAgg*>& agg_exprs,
              const std::vector<std::string>& fields,
              const std::vector<const RexScalar*>& scalar_sources,
              const bool is_agg)
      : filter_expr_(filter_expr),
        target_exprs_(target_exprs),
        group_indices_(group_indices),
        fields_(fields),
        is_agg_(is_agg) {
    CHECK_EQ(fields.size(), target_exprs.size());
    for (auto agg_expr : agg_exprs) {
      agg_exprs_.emplace_back(agg_expr);
    }
    for (auto scalar_source : scalar_sources) {
      scalar_sources_.emplace_back(scalar_source);
    }
  }

  size_t size() const { return target_exprs_.size(); }

  const RexScalar* getFilterExpr() const { return filter_expr_.get(); }

  const Rex* getTargetExpr(const size_t i) const { return target_exprs_[i]; }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  const size_t getScalarSourcesSize() const { return scalar_sources_.size(); }

  const RexScalar* getScalarSource(const size_t i) const { return scalar_sources_[i].get(); }

  const std::vector<size_t>& getGroupIndices() const { return group_indices_; }

  bool isAggregate() const { return is_agg_; }

  std::string toString() const override {
    std::string result = "(RelCompound<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    result += (filter_expr_ ? filter_expr_->toString() : "null") + " ";
    for (const auto target_expr : target_exprs_) {
      result += target_expr->toString() + " ";
    }
    result += "groups: [";
    for (const size_t group_index : group_indices_) {
      result += " " + std::to_string(group_index);
    }
    result += " ] sources: [";
    for (const auto& scalar_source : scalar_sources_) {
      result += " " + scalar_source->toString();
    }
    return result + " ])";
  }

 private:
  const std::unique_ptr<const RexScalar> filter_expr_;
  const std::vector<const Rex*> target_exprs_;
  const std::vector<size_t> group_indices_;
  std::vector<std::unique_ptr<const RexAgg>> agg_exprs_;
  const std::vector<std::string> fields_;
  const bool is_agg_;
  std::vector<std::unique_ptr<const RexScalar>>
      scalar_sources_;  // building blocks for group_indices_ and agg_exprs_; not actually projected, just owned
};

enum class SortDirection { Ascending, Descending };

enum class NullSortedPosition { First, Last };

class SortField {
 public:
  SortField(const size_t field, const SortDirection sort_dir, const NullSortedPosition nulls_pos)
      : field_(field), sort_dir_(sort_dir), nulls_pos_(nulls_pos) {}

 private:
  const size_t field_;
  const SortDirection sort_dir_;
  const NullSortedPosition nulls_pos_;
};

class RelSort : public RelAlgNode {
 public:
  RelSort(const std::vector<SortField>& collation, const RelAlgNode* input) : collation_(collation) {
    inputs_.emplace_back(input);
  }

  std::string toString() const override {
    std::string result = "(RelSort<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
    // TODO(alex)
    return result + ")";
  }

 private:
  const std::vector<SortField> collation_;
};

std::unique_ptr<const RelAlgNode> ra_interpret(const rapidjson::Value&, const Catalog_Namespace::Catalog&);

namespace Analyzer {

class Expr;

}  // namespace Analyzer

std::shared_ptr<Analyzer::Expr> translate_scalar_rex(const RexScalar* rex,
                                                     const int rte_idx,
                                                     const Catalog_Namespace::Catalog& cat);

std::shared_ptr<Analyzer::Expr> translate_aggregate_rex(
    const RexAgg* rex,
    const int rte_idx,
    const Catalog_Namespace::Catalog& cat,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources);

std::string tree_string(const RelAlgNode*, const size_t indent = 0);

#endif  // QUERYENGINE_RELALGABSTRACTINTERPRETER_H
