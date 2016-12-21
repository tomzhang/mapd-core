#include "LeafAggregator.h"
#include "MapDServer.h"
#include "Catalog/Catalog.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/ResultSet.h"
#include "Shared/scope.h"
#include "gen-cpp/MapD.h"
#include <boost/make_shared.hpp>
#include <boost/make_unique.hpp>
#include <glog/logging.h>
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <future>

using apache::thrift::protocol::TBinaryProtocol;
using apache::thrift::transport::TSocket;
using apache::thrift::transport::TTransport;
using apache::thrift::transport::TBufferedTransport;

LeafAggregator::LeafAggregator(const std::vector<LeafHostInfo>& leaves) {
  for (const auto& leaf : leaves) {
    const auto socket = boost::make_shared<TSocket>(leaf.getHost(), leaf.getPort());
    const auto transport = boost::make_shared<TBufferedTransport>(socket);
    transport->open();
    const auto protocol = boost::make_shared<TBinaryProtocol>(transport);
    leaves_.emplace_back(new MapDClient(protocol));
  }
}

#ifdef HAVE_RAVM

namespace {

std::vector<const TableDescriptor*> get_dag_inputs(const RelAlgNode* ra) {
  CHECK(ra);
  const auto scan_ra = dynamic_cast<const RelScan*>(ra);
  if (scan_ra) {
    return {scan_ra->getTableDescriptor()};
  }
  std::vector<const TableDescriptor*> result;
  for (size_t i = 0; i < ra->inputCount(); ++i) {
    const auto in_ra = ra->getInput(i);
    const auto in_ra_inputs = get_dag_inputs(in_ra);
    result.insert(result.end(), in_ra_inputs.begin(), in_ra_inputs.end());
  }
  return result;
}

bool input_is_replicated(const RelAlgNode* ra) {
  CHECK(ra);
  const auto inputs = get_dag_inputs(ra);
  CHECK(!inputs.empty());
  // TODO(alex): warn / throw if neither REPLICATED nor SHARDED are specified
  if (inputs.front()->partitions == "REPLICATED") {
    return true;
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i]->partitions != "REPLICATED") {
      throw std::runtime_error("Join table " + inputs[i]->tableName + " must be replicated");
    }
  }
  if (inputs.front()->partitions != "SHARDED" && inputs.front()->partitions != "REPLICATED") {
    LOG(WARNING) << "Partitioning not properly specified for table '"
                 << inputs.front()->tableName + "', assuming sharded";
  }
  return false;
}

}  // namespace

// TODO(alex): split and clean-up this method
AggregatedResult LeafAggregator::execute(const Catalog_Namespace::SessionInfo& parent_session_info,
                                         const std::string& query_ra) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  const auto& leaf_session_ids = session_it->second;
  std::vector<int64_t> pending_queries;
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    pending_queries.push_back(leaves_[leaf_idx]->start_query(leaf_session_ids[leaf_idx], query_ra));
  }
  CHECK_EQ(leaves_.size(), leaf_session_ids.size());
  bool execution_finished = false;
  unsigned node_id{0};
  TMergeType::type merge_type = TMergeType::REDUCE;
  const auto& cat = parent_session_info.get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId, "", "", 0, 0, nullptr);
  RelAlgExecutor ra_executor(executor.get(), cat);
  const auto ra = deserialize_ra_dag(query_ra, cat, &ra_executor);
  std::mutex leaves_mutex;
  ssize_t crt_subquery_idx = -1;
  const auto subqueries = ra_executor.getSubqueries();
  while (!execution_finished) {
    ++crt_subquery_idx;
    // new execution step
    bool replicated = false;
    if (crt_subquery_idx >= static_cast<ssize_t>(subqueries.size())) {
      CHECK_EQ(static_cast<ssize_t>(subqueries.size()), crt_subquery_idx);
      replicated = input_is_replicated(ra.get());
    } else {
      CHECK_GE(crt_subquery_idx, 0);
      replicated = input_is_replicated(subqueries[crt_subquery_idx]->getRelAlg());
    }
    std::vector<std::shared_ptr<ResultSet>> leaf_results;
    TRowDescriptor row_desc;
    std::vector<TargetInfo> target_infos;
    std::vector<std::future<void>> leaf_futures;
    for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
      if (replicated) {
        TStepResult step_result;
        const auto& leaf = leaves_[leaf_idx];
        leaf->execute_first_step(step_result, pending_queries[leaf_idx]);
        auto result_set = ResultSet::unserialize(step_result.serialized_rows);
        target_infos = result_set->getTargetInfos();
        if (!result_set->definitelyHasNoRows()) {
          leaf_results.emplace_back(result_set.release());
        }
        execution_finished = step_result.execution_finished;
        merge_type = step_result.merge_type;
        row_desc = step_result.row_desc;
        node_id = step_result.node_id;
        break;
      }
      leaf_futures.emplace_back(std::async(std::launch::async,
                                           [&execution_finished,
                                            &leaf_results,
                                            &leaf_session_ids,
                                            &leaves_mutex,
                                            &merge_type,
                                            &node_id,
                                            &pending_queries,
                                            &query_ra,
                                            &row_desc,
                                            &target_infos,
                                            leaf_idx,
                                            this] {
                                             TStepResult step_result;
                                             const auto& leaf = leaves_[leaf_idx];
                                             leaf->execute_first_step(step_result, pending_queries[leaf_idx]);
                                             auto result_set = ResultSet::unserialize(step_result.serialized_rows);
                                             std::lock_guard<std::mutex> lock(leaves_mutex);
                                             target_infos = result_set->getTargetInfos();
                                             if (!result_set->definitelyHasNoRows()) {
                                               leaf_results.emplace_back(result_set.release());
                                             }
                                             if (leaf_idx == 0) {
                                               execution_finished = step_result.execution_finished;
                                               merge_type = step_result.merge_type;
                                               row_desc = step_result.row_desc;
                                               node_id = step_result.node_id;
                                             }
                                           }));
    }
    for (auto& leaf_future : leaf_futures) {
      leaf_future.get();
    }
    QueryMemoryDescriptor empty_query_mem_desc{};
    auto empty_result_set =
        std::make_shared<ResultSet>(target_infos, ExecutorDeviceType::CPU, empty_query_mem_desc, nullptr, nullptr);
    if (merge_type == TMergeType::UNION) {
      CHECK(false);  // TODO(alex)
      continue;
    }
    std::vector<ResultSet*> leaf_results_ptrs;
    for (auto& result : leaf_results) {
      leaf_results_ptrs.push_back(result.get());
    }
    ResultSetManager rs_manager;
    std::shared_ptr<ResultSet> reduced_rs;
    if (leaf_results_ptrs.size() == 1) {
      reduced_rs = leaf_results.front();
    } else if (leaf_results_ptrs.size() > 1) {
      rs_manager.reduce(leaf_results_ptrs);
      auto rs_manager_rs = rs_manager.getOwnResultSet();
      reduced_rs = rs_manager_rs ? rs_manager_rs : leaf_results.front();
    } else {
      reduced_rs = empty_result_set;
    }
    const auto target_meta_infos = target_meta_infos_from_thrift(row_desc);
    AggregatedResult leaves_result{reduced_rs, target_meta_infos};
    CompilationOptions co = {ExecutorDeviceType::CPU, true, ExecutorOptLevel::Default};
    ExecutionOptions eo = {false, true, false, false, true, false, false};
    std::lock_guard<std::mutex> lock(executor->execute_mutex_);
    ScopeGuard row_set_holder = [executor] { executor->row_set_mem_owner_ = nullptr; };
    executor->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
    ScopeGuard restore_input_table_info_cache = [executor] { executor->clearInputTableInfoCache(); };
    if (crt_subquery_idx >= static_cast<ssize_t>(subqueries.size())) {
      CHECK_EQ(static_cast<ssize_t>(subqueries.size()), crt_subquery_idx);
      CHECK(execution_finished);
      ra_executor.addLeafResult(node_id, leaves_result);
      auto ed_list = get_execution_descriptors(ra.get());
      const auto aggregated_result = ra_executor.executeRelAlgSeq(ed_list, co, eo, nullptr, 0);
      const auto& aggregated_result_rows = aggregated_result.getRows();
      auto aggregated_rs =
          aggregated_result_rows.definitelyHasNoRows() ? empty_result_set : aggregated_result_rows.getResultSet();
      CHECK(aggregated_rs);
      return {aggregated_rs, aggregated_result.getTargetsMeta()};
    } else {
      CHECK_GE(crt_subquery_idx, 0);
      CHECK(!execution_finished);
      RelAlgExecutor subquery_executor(executor.get(), cat);
      subquery_executor.addLeafResult(node_id, leaves_result);
      auto current_subquery = subqueries[crt_subquery_idx];
      auto aggregated_result = std::make_shared<ExecutionResult>(
          subquery_executor.executeRelAlgSubQuery(current_subquery->getRelAlg(), co, eo));
      current_subquery->setExecutionResult(aggregated_result);
      broadcastResultSet(reduced_rs.get(), row_desc, pending_queries);
    }
  }
  CHECK(false);
  return {nullptr, {}};
}

#endif  // HAVE_RAVM

void LeafAggregator::connect(const Catalog_Namespace::SessionInfo& parent_session_info,
                             const std::string& user,
                             const std::string& passwd,
                             const std::string& dbname) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(leaf_sessions_mutex_);
  std::vector<TSessionId> leaf_session_ids;
  for (const auto& leaf : leaves_) {
    leaf_session_ids.push_back(leaf->connect(user, passwd, dbname));
  }
  const auto it_ok = leaf_sessions_.emplace(parent_session_info.get_session_id(), leaf_session_ids);
  CHECK(it_ok.second);
}

void LeafAggregator::disconnect(const TSessionId session) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(session);
  const auto& leaf_session_ids = session_it->second;
  CHECK_EQ(leaves_.size(), leaf_session_ids.size());
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    const auto& leaf = leaves_[leaf_idx];
    leaf->disconnect(leaf_session_ids[leaf_idx]);
  }
  leaf_sessions_.erase(session);
}

size_t LeafAggregator::leafCount() const {
  return leaves_.size();
}

void LeafAggregator::broadcastResultSet(const ResultSet* result_set,
                                        const TRowDescriptor& row_desc,
                                        const std::vector<int64_t>& pending_queries) const {
  CHECK_EQ(leaves_.size(), pending_queries.size());
  const auto serialized_result_set = result_set->serialize();
  std::vector<std::future<void>> leaf_futures;
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    leaf_futures.emplace_back(
        std::async(std::launch::async, [&pending_queries, &row_desc, &serialized_result_set, leaf_idx, this] {
          const auto& leaf = leaves_[leaf_idx];
          const auto query_id = pending_queries[leaf_idx];
          leaf->broadcast_serialized_rows(serialized_result_set, row_desc, query_id);
        }));
  }
  for (auto& leaf_future : leaf_futures) {
    leaf_future.get();
  }
}

LeafAggregator::SessionMap::iterator LeafAggregator::getSessionIterator(const TSessionId session) {
  auto session_it = leaf_sessions_.find(session);
  if (session_it == leaf_sessions_.end()) {
    TMapDException ex;
    ex.error_msg = "Session not valid.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  return session_it;
}
