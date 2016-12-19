#include "PendingExecutionClosure.h"

PendingExecutionClosure::PendingExecutionClosure(std::shared_ptr<const RelAlgNode> ra,
                                                 const int64_t id,
                                                 std::unique_ptr<RelAlgExecutor>& ra_executor,
                                                 const Catalog_Namespace::Catalog& cat,
                                                 const RelAlgExecutionOptions& rel_alg_eo)
    : ra_(ra), id_(id), crt_ed_list_idx_(-1), ra_executor_(std::move(ra_executor)), rel_alg_eo_(rel_alg_eo) {
  ed_list_ = get_execution_descriptors(ra.get());
}

PendingExecutionClosure* PendingExecutionClosure::create(std::shared_ptr<const RelAlgNode> ra,
                                                         std::unique_ptr<RelAlgExecutor>& ra_executor,
                                                         const Catalog_Namespace::Catalog& cat,
                                                         const RelAlgExecutionOptions& rel_alg_eo) {
  std::lock_guard<std::mutex> pending_queries_lock(pending_queries_mutex_);
  const auto it_ok = pending_queries_.emplace(pending_query_next_id_,
                                              std::unique_ptr<PendingExecutionClosure>(new PendingExecutionClosure(
                                                  ra, pending_query_next_id_, ra_executor, cat, rel_alg_eo)));
  ++pending_query_next_id_;
  CHECK(it_ok.second);
  return (*it_ok.first).second.get();
}

bool PendingExecutionClosure::executeNextStep(const int64_t query_id) {
  PendingExecutionClosure* closure{nullptr};
  {
    std::lock_guard<std::mutex> pending_queries_lock(pending_queries_mutex_);
    const auto it = pending_queries_.find(query_id);
    CHECK(it != pending_queries_.end());
    closure = it->second.get();
  }
  CHECK(closure);
  const bool done = closure->executeNextStep();
  if (done) {
    std::lock_guard<std::mutex> pending_queries_lock(pending_queries_mutex_);
    pending_queries_.erase(closure->id_);
  }
  return done;
}

int64_t PendingExecutionClosure::getId() const {
  return id_;
}

bool PendingExecutionClosure::executeNextStep() {
  ++crt_ed_list_idx_;
  ra_executor_->executeRelAlgStep(
      crt_ed_list_idx_, ed_list_, rel_alg_eo_.co, rel_alg_eo_.eo, rel_alg_eo_.render_info, rel_alg_eo_.queue_time_ms);
  return static_cast<size_t>(crt_ed_list_idx_ + 1) == ed_list_.size();  // done?
}

std::unordered_map<int64_t, std::unique_ptr<PendingExecutionClosure>> PendingExecutionClosure::pending_queries_;
int64_t PendingExecutionClosure::pending_query_next_id_{0};
std::mutex PendingExecutionClosure::pending_queries_mutex_;
