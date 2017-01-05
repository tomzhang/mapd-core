#include "PendingExecutionClosure.h"

#include <boost/algorithm/string/join.hpp>

PendingExecutionClosure::PendingExecutionClosure(std::shared_ptr<const RelAlgNode> ra,
                                                 const int64_t id,
                                                 std::unique_ptr<RelAlgExecutor>& ra_executor,
                                                 const Catalog_Namespace::Catalog& cat,
                                                 const RelAlgExecutionOptions& rel_alg_eo)
    : ra_(ra),
      id_(id),
      crt_subquery_idx_(-1),
      ra_executor_(std::move(ra_executor)),
      rel_alg_eo_(rel_alg_eo),
      col_range_cache_(ra_executor_->computeColRangesCache(ra.get())) {}

PendingExecutionClosure* PendingExecutionClosure::create(std::shared_ptr<const RelAlgNode> ra,
                                                         std::unique_ptr<RelAlgExecutor>& ra_executor,
                                                         const Catalog_Namespace::Catalog& cat,
                                                         const RelAlgExecutionOptions& rel_alg_eo) {
  std::lock_guard<std::mutex> pending_queries_lock(pending_queries_mutex_);
  if (!pending_queries_.empty()) {
    std::vector<std::string> orphaned_ids;
    for (const auto& kv : pending_queries_) {
      orphaned_ids.push_back(std::to_string(kv.first));
    }
    LOG(INFO) << "Orphaned queries " << boost::algorithm::join(orphaned_ids, ", ")
              << " cancelled, most likely because of an aggregator crash";
    decltype(pending_queries_)().swap(pending_queries_);
  }
  const auto it_ok = pending_queries_.emplace(pending_query_next_id_,
                                              std::unique_ptr<PendingExecutionClosure>(new PendingExecutionClosure(
                                                  ra, pending_query_next_id_, ra_executor, cat, rel_alg_eo)));
  ++pending_query_next_id_;
  CHECK(it_ok.second);
  return (*it_ok.first).second.get();
}

FirstStepExecutionResult PendingExecutionClosure::executeNextStep(const int64_t query_id,
                                                                  const AggregatedColRange& col_ranges) {
  PendingExecutionClosure* closure{nullptr};
  {
    std::lock_guard<std::mutex> pending_queries_lock(pending_queries_mutex_);
    const auto it = pending_queries_.find(query_id);
    CHECK(it != pending_queries_.end());
    closure = it->second.get();
  }
  CHECK(closure);
  const auto result = closure->executeNextStep(col_ranges);
  if (result.is_outermost_query) {
    std::lock_guard<std::mutex> pending_queries_lock(pending_queries_mutex_);
    pending_queries_.erase(closure->id_);
  }
  return result;
}

void PendingExecutionClosure::setCurrentSubqueryResult(const int64_t query_id,
                                                       const std::shared_ptr<const ExecutionResult> result) {
  PendingExecutionClosure* closure{nullptr};
  {
    std::lock_guard<std::mutex> pending_queries_lock(pending_queries_mutex_);
    const auto it = pending_queries_.find(query_id);
    CHECK(it != pending_queries_.end());
    closure = it->second.get();
  }
  CHECK(closure);
  closure->setCurrentSubqueryResult(result);
}

int64_t PendingExecutionClosure::getId() const {
  return id_;
}

const AggregatedColRange& PendingExecutionClosure::getColRangeCache() const {
  return col_range_cache_;
}

FirstStepExecutionResult PendingExecutionClosure::executeNextStep(const AggregatedColRange& col_ranges) {
  ++crt_subquery_idx_;
  if (crt_subquery_idx_ == 0) {
    ra_executor_->prepareLeafExecution(col_ranges);
  }
  const auto subqueries = ra_executor_->getSubqueries();
  if (crt_subquery_idx_ >= static_cast<ssize_t>(subqueries.size())) {
    CHECK_EQ(static_cast<ssize_t>(subqueries.size()), crt_subquery_idx_);
    auto result =
        ra_executor_->executeRelAlgQueryFirstStep(ra_.get(), rel_alg_eo_.co, rel_alg_eo_.eo, rel_alg_eo_.render_info);
    result.is_outermost_query = true;
    return result;
  }
  return ra_executor_->executeRelAlgQueryFirstStep(
      subqueries[crt_subquery_idx_]->getRelAlg(), rel_alg_eo_.co, rel_alg_eo_.eo, rel_alg_eo_.render_info);
}

void PendingExecutionClosure::setCurrentSubqueryResult(const std::shared_ptr<const ExecutionResult> result) {
  CHECK_GE(crt_subquery_idx_, 0);
  const auto subqueries = ra_executor_->getSubqueries();
  CHECK_LT(crt_subquery_idx_, static_cast<ssize_t>(subqueries.size()));
  subqueries[crt_subquery_idx_]->setExecutionResult(result);
}

std::unordered_map<int64_t, std::unique_ptr<PendingExecutionClosure>> PendingExecutionClosure::pending_queries_;
int64_t PendingExecutionClosure::pending_query_next_id_{0};
std::mutex PendingExecutionClosure::pending_queries_mutex_;
