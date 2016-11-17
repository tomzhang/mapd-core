#include "LeafAggregator.h"
#include "Catalog/Catalog.h"
#include "QueryEngine/ResultSet.h"
#include "gen-cpp/MapD.h"
#include <boost/make_shared.hpp>
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

namespace {

// placeholder, we'll implement it in the ResultSet module
ResultSet* unserialize_result_set(const std::string& serialized_results) {
  CHECK(false);
  return nullptr;
}

// placeholder, we'll implement it in the ResultSet module
std::string serialize_result_set(const ResultSet* result_set) {
  CHECK(false);
  return "";
}

}  // namespace

void LeafAggregator::execute(TQueryResult& _return,
                             const Catalog_Namespace::SessionInfo& parent_session_info,
                             const std::string& query_str,
                             const bool column_format,
                             const std::string& nonce) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  const auto& leaf_session_ids = session_it->second;
  CHECK_EQ(leaves_.size(), leaf_session_ids.size());
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    pending_queries_[leaf_idx] =
        leaves_[leaf_idx]->start_query(leaf_session_ids[leaf_idx], query_str, column_format, nonce);
  }
  bool execution_finished = false;
  TMergeType::type merge_type = TMergeType::REDUCE;
  bool sharded = true;
  while (!execution_finished) {
    // new execution step
    std::vector<std::unique_ptr<ResultSet>> leaf_results;
    for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
      const auto& leaf = leaves_[leaf_idx];
      TStepResult step_result;
      leaf->execute_step(step_result, pending_queries_[leaf_idx]);
      leaf_results.emplace_back(unserialize_result_set(step_result.serialized_rows));
      if (leaf_idx == 0) {
        execution_finished = step_result.execution_finished;
        merge_type = step_result.merge_type;
        sharded = step_result.sharded;
      } else {
        // leaves move in lock-step
        CHECK_EQ(execution_finished, step_result.execution_finished);
        CHECK_EQ(merge_type, step_result.merge_type);
        CHECK_EQ(sharded, step_result.sharded);
      }
    }
    if (!sharded) {
      CHECK(!leaf_results.empty());
      broadcastResultSet(leaf_results.front().get());  // TODO(alex): avoid, most of the time should be possible
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
    const auto reduced_rs = rs_manager.reduce(leaf_results_ptrs);
    if (execution_finished) {
      // set results in _return
      CHECK(false);
    } else {
      broadcastResultSet(reduced_rs);
    }
  }
  CHECK(false);
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
  const auto serialized_result_set = serialize_result_set(result_set);
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
