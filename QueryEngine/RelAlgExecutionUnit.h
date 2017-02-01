/**
 * @file    RelAlgExecutionUnit.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Execution unit for relational algebra. It's a low-level description
 *          of any relational algebra operation in a format understood by our VM.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_RELALGEXECUTIONUNIT_H
#define QUERYENGINE_RELALGEXECUTIONUNIT_H

#include "InputDescriptors.h"
#include "../Shared/sqldefs.h"

#include <list>
#include <memory>
#include <vector>

enum class SortAlgorithm { Default, SpeculativeTopN };

namespace Analyzer {

class Expr;
class NDVEstimator;
struct OrderEntry;

}  // Analyzer

struct SortInfo {
  const std::list<Analyzer::OrderEntry> order_entries;
  const SortAlgorithm algorithm;
  const size_t limit;
  const size_t offset;
};

struct RelAlgExecutionUnit {
  const std::vector<InputDescriptor> input_descs;
  const std::vector<InputDescriptor> extra_input_descs;
  const std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  const std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> quals;
  const JoinType join_type;
  const std::vector<std::pair<int, size_t>> join_dimensions;
  const std::list<std::shared_ptr<Analyzer::Expr>> inner_join_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> outer_join_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  std::vector<Analyzer::Expr*> target_exprs;
  const std::vector<Analyzer::Expr*> orig_target_exprs;
  const std::shared_ptr<Analyzer::NDVEstimator> estimator;
  const SortInfo sort_info;
  size_t scan_limit;
};

#endif  // QUERYENGINE_RELALGEXECUTIONUNIT_H
