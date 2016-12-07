#include "LeafAggregator.h"
#include "MapDServer.h"
#include "Catalog/Catalog.h"
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

AggregatedResult LeafAggregator::execute(const Catalog_Namespace::SessionInfo& parent_session_info,
                                         const std::string& query_str,
                                         const bool column_format,
                                         const std::string& nonce) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  const auto& leaf_session_ids = session_it->second;
  CHECK_EQ(leaves_.size(), leaf_session_ids.size());
  bool execution_finished = false;
  TMergeType::type merge_type = TMergeType::REDUCE;
  bool sharded = true;
  while (!execution_finished) {
    // new execution step
    std::vector<std::unique_ptr<ResultSet>> leaf_results;
    TRowDescriptor row_desc;
    for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
      const auto& leaf = leaves_[leaf_idx];
      TStepResult step_result;
      leaf->execute_first_step(step_result, leaf_session_ids[leaf_idx], query_str, column_format, nonce);
      auto result_set = ResultSet::unserialize(step_result.serialized_rows);
      if (!result_set->definitelyHasNoRows()) {
        leaf_results.emplace_back(std::move(result_set));
      }
      if (leaf_idx == 0) {
        execution_finished = step_result.execution_finished;
        merge_type = step_result.merge_type;
        sharded = step_result.sharded;
        row_desc = step_result.row_desc;
      } else {
        // leaves move in lock-step
        CHECK_EQ(execution_finished, step_result.execution_finished);
        CHECK_EQ(merge_type, step_result.merge_type);
        CHECK_EQ(sharded, step_result.sharded);
      }
    }
    QueryMemoryDescriptor empty_query_mem_desc{};
    std::vector<TargetInfo> empty_target_infos;
    auto empty_result_set = boost::make_unique<ResultSet>(
        empty_target_infos, ExecutorDeviceType::CPU, empty_query_mem_desc, nullptr, nullptr);
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
    if (leaf_results.empty()) {
      if (execution_finished) {
        return {std::move(empty_result_set), {}};
      } else {
        broadcastResultSet(empty_result_set.get());
      }
      continue;
    }
    std::vector<ResultSet*> leaf_results_ptrs;
    for (auto& result : leaf_results) {
      leaf_results_ptrs.push_back(result.get());
    }
    ResultSetManager rs_manager;
    const auto reduced_rs = rs_manager.reduce(leaf_results_ptrs);
    if (execution_finished) {
      auto& rs_manager_rs = rs_manager.getOwnResultSet();
      return {rs_manager_rs ? std::move(rs_manager_rs) : std::move(leaf_results.front()),
              target_meta_infos_from_thrift(row_desc)};
    } else {
      broadcastResultSet(reduced_rs);
    }
  }
  CHECK(false);
  return {nullptr, {}};
}

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
