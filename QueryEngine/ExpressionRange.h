#ifndef QUERYENGINE_EXPRESSIONRANGE_H
#define QUERYENGINE_EXPRESSIONRANGE_H

#include "../Analyzer/Analyzer.h"
#include "../Fragmenter/Fragmenter.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <deque>

typedef boost::multiprecision::number<
    boost::multiprecision::
        cpp_int_backend<64, 64, boost::multiprecision::signed_magnitude, boost::multiprecision::checked, void>>
    checked_int64_t;

enum class ExpressionRangeType {
  Invalid,
  Integer,
  FloatingPoint,
};

class ExpressionRange;

template <typename T>
T getMin(const ExpressionRange& other);

template <typename T>
T getMax(const ExpressionRange& other);

class ExpressionRange {
 public:
  static ExpressionRange makeIntRange(const int64_t int_min,
                                      const int64_t int_max,
                                      const int64_t bucket,
                                      const bool has_nulls) {
    return ExpressionRange(int_min, int_max, bucket, has_nulls);
  }

  static ExpressionRange makeFpRange(const double fp_min, const double fp_max, const bool has_nulls) {
    return ExpressionRange(fp_min, fp_max, has_nulls);
  }

  static ExpressionRange makeInvalidRange() { return ExpressionRange(); }

  int64_t getIntMin() const {
    CHECK(ExpressionRangeType::Integer == type_);
    return int_min_;
  }

  int64_t getIntMax() const {
    CHECK(ExpressionRangeType::Integer == type_);
    return int_max_;
  }

  double getFpMin() const {
    CHECK(ExpressionRangeType::FloatingPoint == type_);
    return fp_min_;
  }

  double getFpMax() const {
    CHECK(ExpressionRangeType::FloatingPoint == type_);
    return fp_max_;
  }

  ExpressionRangeType getType() const { return type_; }

  int64_t getBucket() const { return bucket_; }

  bool hasNulls() const { return has_nulls_; }

  ExpressionRange operator+(const ExpressionRange& other) const;
  ExpressionRange operator-(const ExpressionRange& other) const;
  ExpressionRange operator*(const ExpressionRange& other) const;
  ExpressionRange operator/(const ExpressionRange& other) const;
  ExpressionRange operator||(const ExpressionRange& other) const;

 private:
  ExpressionRange(const int64_t int_min_in, const int64_t int_max_in, const int64_t bucket, const bool has_nulls_in)
      : type_(ExpressionRangeType::Integer),
        has_nulls_(has_nulls_in),
        int_min_(int_min_in),
        int_max_(int_max_in),
        bucket_(bucket) {}

  ExpressionRange(const double fp_min_in, const double fp_max_in, const bool has_nulls_in)
      : type_(ExpressionRangeType::FloatingPoint),
        has_nulls_(has_nulls_in),
        fp_min_(fp_min_in),
        fp_max_(fp_max_in),
        bucket_(0) {}

  ExpressionRange() : type_(ExpressionRangeType::Invalid), has_nulls_(false), bucket_(0) {}

  template <class T, class BinOp>
  ExpressionRange binOp(const ExpressionRange& other, const BinOp& bin_op) const {
    if (type_ == ExpressionRangeType::Invalid || other.type_ == ExpressionRangeType::Invalid) {
      return ExpressionRange::makeInvalidRange();
    }
    try {
      std::vector<T> limits{bin_op(getMin<T>(*this), getMin<T>(other)),
                            bin_op(getMin<T>(*this), getMax<T>(other)),
                            bin_op(getMax<T>(*this), getMin<T>(other)),
                            bin_op(getMax<T>(*this), getMax<T>(other))};
      ExpressionRange result;
      result.type_ = (type_ == ExpressionRangeType::Integer && other.type_ == ExpressionRangeType::Integer)
                         ? ExpressionRangeType::Integer
                         : ExpressionRangeType::FloatingPoint;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      switch (result.type_) {
        case ExpressionRangeType::Integer: {
          result.int_min_ = *std::min_element(limits.begin(), limits.end());
          result.int_max_ = *std::max_element(limits.begin(), limits.end());
          break;
        }
        case ExpressionRangeType::FloatingPoint: {
          result.fp_min_ = *std::min_element(limits.begin(), limits.end());
          result.fp_max_ = *std::max_element(limits.begin(), limits.end());
          break;
        }
        default:
          CHECK(false);
      }
      return result;
    } catch (...) {
      return ExpressionRange::makeInvalidRange();
    }
  }

  ExpressionRangeType type_;
  bool has_nulls_;
  union {
    int64_t int_min_;
    double fp_min_;
  };
  union {
    int64_t int_max_;
    double fp_max_;
  };
  int64_t bucket_;
};

template <>
inline int64_t getMin<int64_t>(const ExpressionRange& e) {
  return e.getIntMin();
}

template <>
inline double getMin<double>(const ExpressionRange& e) {
  return e.getFpMin();
}

template <>
inline int64_t getMax<int64_t>(const ExpressionRange& e) {
  return e.getIntMax();
}

template <>
inline double getMax<double>(const ExpressionRange& e) {
  return e.getFpMax();
}

class Executor;

ExpressionRange getExpressionRange(const Analyzer::Expr*,
                                   const std::vector<Fragmenter_Namespace::TableInfo>&,
                                   const Executor*);

#endif  // QUERYENGINE_EXPRESSIONRANGE_H
