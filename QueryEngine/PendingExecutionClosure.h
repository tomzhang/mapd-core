/*
 * @file    PendingExecutionClosure.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Keeps track of current execution stage (and other state),
 *          needed for lock-step distributed query execution.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */
#ifndef QUERYENGINE_PENDINGEXECUTIONCLOSURE_H
#define QUERYENGINE_PENDINGEXECUTIONCLOSURE_H

#include "RelAlgExecutor.h"

#include <mutex>
#include <unordered_map>

struct RelAlgExecutionOptions {
  CompilationOptions co;
  ExecutionOptions eo;
  RenderInfo* render_info;
  int64_t queue_time_ms;
};

class PendingExecutionClosure {
 public:
  static PendingExecutionClosure* create(std::shared_ptr<const RelAlgNode> ra,
                                         std::unique_ptr<RelAlgExecutor>& ra_executor,
                                         const Catalog_Namespace::Catalog& cat,
                                         const RelAlgExecutionOptions& rel_alg_eo);

  static FirstStepExecutionResult executeNextStep(const int64_t query_id, const AggregatedColRange& col_ranges);

  static void setCurrentSubqueryResult(const int64_t query_id, const std::shared_ptr<const ExecutionResult> result);

  int64_t getId() const;

  const AggregatedColRange& getColRangeCache() const;

 private:
  PendingExecutionClosure(std::shared_ptr<const RelAlgNode> ra,
                          const int64_t id,
                          std::unique_ptr<RelAlgExecutor>& ra_executor,
                          const Catalog_Namespace::Catalog& cat,
                          const RelAlgExecutionOptions& rel_alg_eo);

  FirstStepExecutionResult executeNextStep(const AggregatedColRange& col_ranges);

  void setCurrentSubqueryResult(const std::shared_ptr<const ExecutionResult> result);

  const std::shared_ptr<const RelAlgNode> ra_;
  const int64_t id_;
  ssize_t crt_subquery_idx_;
  std::unique_ptr<RelAlgExecutor> ra_executor_;
  RelAlgExecutionOptions rel_alg_eo_;
  AggregatedColRange col_range_cache_;

  static std::unordered_map<int64_t, std::unique_ptr<PendingExecutionClosure>> pending_queries_;
  static int64_t pending_query_next_id_;
  static std::mutex pending_queries_mutex_;
};

#endif  // QUERYENGINE_PENDINGEXECUTIONCLOSURE_H
