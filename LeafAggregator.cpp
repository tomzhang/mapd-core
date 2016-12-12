#include "LeafAggregator.h"
#include "MapDServer.h"
#include "Catalog/Catalog.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/ResultSet.h"
#include "gen-cpp/MapD.h"
#include <boost/make_shared.hpp>
#include <boost/make_unique.hpp>
#include <glog/logging.h>
#include <thrift/transport/TSocket.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

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
  pending_queries_.resize(leaves.size());
}

std::vector<TargetMetaInfo> target_meta_infos_from_thrift(const TRowDescriptor& row_desc) {
  std::vector<TargetMetaInfo> target_meta_infos;
  for (const auto& col : row_desc) {
    target_meta_infos.emplace_back(col.col_name, type_info_from_thrift(col.col_type));
  }
  return target_meta_infos;
}

#ifdef HAVE_RAVM
AggregatedResult LeafAggregator::execute(const Catalog_Namespace::SessionInfo& parent_session_info,
                                         const std::string& query_ra,
                                         const bool column_format,
                                         const std::string& nonce) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  const auto& leaf_session_ids = session_it->second;
  CHECK_EQ(leaves_.size(), leaf_session_ids.size());
  bool execution_finished = false;
  unsigned node_id{0};
  TMergeType::type merge_type = TMergeType::REDUCE;
  bool sharded = true;
  const auto& cat = parent_session_info.get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId, "", "", 0, 0, nullptr);
  RelAlgExecutor ra_executor(executor.get(), cat);
  while (!execution_finished) {
    // new execution step
    std::vector<std::shared_ptr<ResultSet>> leaf_results;
    TRowDescriptor row_desc;
    std::vector<TargetInfo> target_infos;
    for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
      const auto& leaf = leaves_[leaf_idx];
      TStepResult step_result;
      leaf->execute_first_step(step_result, leaf_session_ids[leaf_idx], query_ra, column_format, nonce);
      auto result_set = ResultSet::unserialize(step_result.serialized_rows);
      target_infos = result_set->getTargetInfos();
      if (!result_set->definitelyHasNoRows()) {
        leaf_results.emplace_back(result_set.release());
      }
      if (leaf_idx == 0) {
        execution_finished = step_result.execution_finished;
        merge_type = step_result.merge_type;
        sharded = step_result.sharded;
        row_desc = step_result.row_desc;
        node_id = step_result.node_id;
      } else {
        // leaves move in lock-step
        CHECK_EQ(execution_finished, step_result.execution_finished);
        CHECK_EQ(merge_type, step_result.merge_type);
        CHECK_EQ(sharded, step_result.sharded);
        CHECK_EQ(node_id, step_result.node_id);
      }
    }
    QueryMemoryDescriptor empty_query_mem_desc{};
    auto empty_result_set =
        std::make_shared<ResultSet>(target_infos, ExecutorDeviceType::CPU, empty_query_mem_desc, nullptr, nullptr);
    if (!sharded) {
      broadcastResultSet(leaf_results.empty()
                             ? empty_result_set.get()
                             : leaf_results.front().get());  // TODO(alex): avoid, most of the time should be possible
      continue;
    }
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
    if (!leaf_results_ptrs.empty()) {
      rs_manager.reduce(leaf_results_ptrs);
      auto rs_manager_rs = rs_manager.getOwnResultSet();
      reduced_rs = rs_manager_rs ? rs_manager_rs : leaf_results.front();
    } else {
      reduced_rs = empty_result_set;
    }
    if (execution_finished) {
      const auto target_meta_infos = target_meta_infos_from_thrift(row_desc);
      AggregatedResult leaves_result{reduced_rs, target_meta_infos};
      ra_executor.addLeafResult(node_id, leaves_result);
      CompilationOptions co = {ExecutorDeviceType::CPU, true, ExecutorOptLevel::Default};
      ExecutionOptions eo = {false, true, false, false, true, false};
      const auto aggregated_result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
      const auto& aggregated_result_rows = aggregated_result.getRows();
      auto aggregated_rs =
          aggregated_result_rows.definitelyHasNoRows() ? empty_result_set : aggregated_result_rows.getResultSet();
      CHECK(aggregated_rs);
      return {aggregated_rs, aggregated_result.getTargetsMeta()};
    } else {
      broadcastResultSet(reduced_rs.get());
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

void LeafAggregator::broadcastResultSet(const ResultSet* result_set) const {
  const auto serialized_result_set = result_set->serialize();
  for (const auto& leaf : leaves_) {
    leaf->broadcast_serialized_rows(serialized_result_set);
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
