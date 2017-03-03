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
using apache::thrift::TException;
using apache::thrift::transport::TTransportException;

PersistentLeafClient::PersistentLeafClient(const LeafHostInfo& leaf_host) noexcept : leaf_host_(leaf_host) {
  try {
    setupClient();
  } catch (const TTransportException&) {
    LOG(INFO) << "Couldn't connect to leaf " << leaf_host.getHost() << ":" << leaf_host.getPort();
  }
}

TSessionId PersistentLeafClient::connect(const std::string& user,
                                         const std::string& passwd,
                                         const std::string& dbname) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  try {
    return client_->connect(user, passwd, dbname);
  } catch (const TTransportException&) {
    setupClient();
  }
  return client_->connect(user, passwd, dbname);
}

void PersistentLeafClient::disconnect(const TSessionId session) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->disconnect(session);
}

void PersistentLeafClient::interrupt(const TSessionId session) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->interrupt(session);
}

void PersistentLeafClient::start_query(TPendingQuery& _return, const TSessionId session, const std::string& query_ra) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->start_query(_return, session, query_ra);
}

void PersistentLeafClient::execute_first_step(TStepResult& _return, const TPendingQuery& pending_query) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->execute_first_step(_return, pending_query);
}

void PersistentLeafClient::broadcast_serialized_rows(const std::string& serialized_rows,
                                                     const TRowDescriptor& row_desc,
                                                     const TQueryId query_id) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->broadcast_serialized_rows(serialized_rows, row_desc, query_id);
}

void PersistentLeafClient::render_vega(TRenderResult& _return,
                                       const TSessionId session,
                                       const int64_t widget_id,
                                       const std::string& vega_json,
                                       const int compressionLevel,
                                       const std::string& nonce) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->render_vega(_return, session, widget_id, vega_json, compressionLevel, nonce);
}

void PersistentLeafClient::get_result_row_for_pixel(
    TPixelTableRowResult& _return,
    const TSessionId session,
    const int64_t widget_id,
    const TPixel& pixel,
    const std::map<std::string, std::vector<std::string>>& table_col_names,
    const bool column_format,
    const int32_t pixelRadius,
    const std::string& nonce) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->get_result_row_for_pixel(
      _return, session, widget_id, pixel, table_col_names, column_format, pixelRadius, nonce);
}

void PersistentLeafClient::sql_execute(TQueryResult& _return,
                                       const TSessionId session,
                                       const std::string& query_str,
                                       const bool column_format,
                                       const std::string& nonce) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  setupClientIfNull();
  client_->sql_execute(_return, session, query_str, column_format, nonce);
}

void PersistentLeafClient::setupClient() {
  const auto socket = boost::make_shared<TSocket>(leaf_host_.getHost(), leaf_host_.getPort());
  socket->setConnTimeout(5000);
  socket->setRecvTimeout(10000);
  socket->setSendTimeout(10000);
  const auto transport = boost::make_shared<TBufferedTransport>(socket);
  transport->open();
  const auto protocol = boost::make_shared<TBinaryProtocol>(transport);
  client_.reset(new MapDClient(protocol));
}

void PersistentLeafClient::setupClientIfNull() {
  if (!client_) {
    setupClient();
  }
}

LeafAggregator::LeafAggregator(const std::vector<LeafHostInfo>& leaves) {
  for (const auto& leaf : leaves) {
    leaves_.emplace_back(new PersistentLeafClient(leaf));
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

bool table_is_replicated(const TableDescriptor* td) {
  return td->partitions == "REPLICATED";
}

void check_replication_constraints(const RelAlgNode* ra) {
  CHECK(ra);
  const auto scan = dynamic_cast<const RelScan*>(ra);
  if (scan) {
    return;
  }
  const auto join = dynamic_cast<const RelJoin*>(ra);
  if (!join) {
    CHECK_EQ(size_t(1), ra->inputCount());
    check_replication_constraints(ra->getInput(0));
    return;
  }
  for (size_t i = 0; i < join->inputCount(); ++i) {
    const auto input = ra->getInput(i);
    CHECK(input);
    const auto input_scan = dynamic_cast<const RelScan*>(input);
    if (!input_scan) {
      check_replication_constraints(input);
    } else {
      const auto td = input_scan->getTableDescriptor();
      if (td->partitions != "SHARDED" && td->partitions != "REPLICATED") {
        LOG(WARNING) << "Partitioning not properly specified for table '" << td->tableName + "', assuming sharded";
      }
      if (i > 0) {
        if (!table_is_replicated(td)) {
          throw std::runtime_error("Join table " + td->tableName + " must be replicated");
        }
      }
    }
  }
}

bool input_is_replicated(const RelAlgNode* ra) {
  CHECK(ra);
  const auto inputs = get_dag_inputs(ra);
  CHECK(!inputs.empty());
  return table_is_replicated(inputs.front());
}

typedef std::vector<TTableGeneration> TTableGenerations;

const TTableGeneration& find_table_generation(const TTableGenerations& table_generations, const int table_id) {
  const auto it = std::find_if(table_generations.begin(),
                               table_generations.end(),
                               [table_id](const TTableGeneration& needle) { return table_id == needle.table_id; });
  CHECK(it != table_generations.end());
  return *it;
}

typedef std::vector<TColumnRange> TColumnRanges;

const TColumnRange& find_column_range(const TColumnRanges& column_ranges, const int col_id, const int table_id) {
  const auto it = std::find_if(column_ranges.begin(), column_ranges.end(), [col_id, table_id](const TColumnRange& r) {
    return r.col_id == col_id && r.table_id == table_id;
  });
  CHECK(it != column_ranges.end());
  return *it;
}

TColumnRanges aggregate_two_leaf_ranges(const TColumnRanges& acc_ranges,
                                        const size_t leaf_idx,
                                        const Catalog_Namespace::Catalog& cat,
                                        const std::vector<TPendingQuery>& pending_queries) {
  CHECK_GT(leaf_idx, 0);
  CHECK_LT(leaf_idx, pending_queries.size());
  const auto& leaf_ranges = pending_queries[leaf_idx].column_ranges;
  CHECK_EQ(acc_ranges.size(), leaf_ranges.size());
  TColumnRanges result;
  for (const auto& lhs_column_range : acc_ranges) {
    const auto it = std::find_if(leaf_ranges.begin(), leaf_ranges.end(), [&lhs_column_range](const TColumnRange& r) {
      return r.col_id == lhs_column_range.col_id && r.table_id == lhs_column_range.table_id;
    });
    CHECK(it != leaf_ranges.end());
    const auto& rhs_column_range = *it;
    CHECK_EQ(lhs_column_range.type, rhs_column_range.type);
    const auto td = cat.getMetadataForTable(lhs_column_range.table_id);
    CHECK(td);
    if (table_is_replicated(td)) {
      result.push_back(
          find_column_range(pending_queries[0].column_ranges, lhs_column_range.col_id, lhs_column_range.table_id));
      continue;
    }
    TColumnRange column_range;
    column_range.type = lhs_column_range.type;
    column_range.col_id = lhs_column_range.col_id;
    column_range.table_id = lhs_column_range.table_id;
    switch (column_range.type) {
      case TExpressionRangeType::INTEGER: {
        // handle empty lhs range
        if (lhs_column_range.int_min > lhs_column_range.int_max) {
          column_range = rhs_column_range;
          break;
        }
        // handle empty rhs range
        if (rhs_column_range.int_min > rhs_column_range.int_max) {
          column_range = lhs_column_range;
          break;
        }
        CHECK_EQ(lhs_column_range.bucket, rhs_column_range.bucket);
        column_range.bucket = lhs_column_range.bucket;
        int64_t start_rowid{0};
        const auto cd = cat.getMetadataForColumn(column_range.table_id, column_range.col_id);
        CHECK(cd);
        if (cd && cd->isVirtualCol) {
          CHECK(cd->columnName == "rowid");
          CHECK_EQ(kBIGINT, cd->columnType.get_type());
          const auto& table_generation =
              find_table_generation(pending_queries[leaf_idx].table_generations, column_range.table_id);
          const auto& acc_table_generation =
              find_table_generation(pending_queries[0].table_generations, column_range.table_id);
          // The first leaf rowid's always start from 0, makes this range aggregation logic easier.
          CHECK_EQ(0, acc_table_generation.start_rowid);
          start_rowid = table_generation.start_rowid;
        }
        column_range.int_min = std::min(lhs_column_range.int_min, rhs_column_range.int_min + start_rowid);
        column_range.int_max = std::max(lhs_column_range.int_max, rhs_column_range.int_max + start_rowid);
        column_range.has_nulls = lhs_column_range.has_nulls || rhs_column_range.has_nulls;
        break;
      }
      case TExpressionRangeType::FLOAT:
      case TExpressionRangeType::DOUBLE:
        column_range.fp_min = std::min(lhs_column_range.fp_min, rhs_column_range.fp_min);
        column_range.fp_max = std::max(lhs_column_range.fp_max, rhs_column_range.fp_max);
        column_range.has_nulls = lhs_column_range.has_nulls || rhs_column_range.has_nulls;
        break;
      case TExpressionRangeType::INVALID:
        break;
      default:
        CHECK(false);
    }
    result.push_back(column_range);
  }
  return result;
}

TColumnRanges aggregate_leaf_ranges(const Catalog_Namespace::Catalog& cat,
                                    const std::vector<TPendingQuery>& pending_queries) {
  CHECK(!pending_queries.empty());
  auto aggregated_ranges = pending_queries.front().column_ranges;
  for (size_t i = 1; i < pending_queries.size(); ++i) {
    aggregated_ranges = aggregate_two_leaf_ranges(aggregated_ranges, i, cat, pending_queries);
  }
  return aggregated_ranges;
}

typedef std::vector<TDictionaryGeneration> TDictionaryGenerations;

TDictionaryGenerations aggregate_two_leaf_dictionary_generations(const TDictionaryGenerations& lhs,
                                                                 const TDictionaryGenerations& rhs) {
  CHECK_EQ(lhs.size(), rhs.size());
  TDictionaryGenerations result;
  for (const auto& lhs_dictionary_generation : lhs) {
    const auto it = std::find_if(rhs.begin(), rhs.end(), [&lhs_dictionary_generation](const TDictionaryGeneration& r) {
      return r.dict_id == lhs_dictionary_generation.dict_id;
    });
    CHECK(it != rhs.end());
    const auto& rhs_dictionary_generation = *it;
    TDictionaryGeneration dictionary_generation;
    dictionary_generation.dict_id = lhs_dictionary_generation.dict_id;
    dictionary_generation.entry_count =
        std::min(lhs_dictionary_generation.entry_count, rhs_dictionary_generation.entry_count);
    result.push_back(dictionary_generation);
  }
  return result;
}

TDictionaryGenerations aggregate_dictionary_generations(const std::vector<TPendingQuery>& pending_queries) {
  CHECK(!pending_queries.empty());
  auto dictionary_generations = pending_queries.front().dictionary_generations;
  for (size_t i = 1; i < pending_queries.size(); ++i) {
    dictionary_generations =
        aggregate_two_leaf_dictionary_generations(dictionary_generations, pending_queries[i].dictionary_generations);
  }
  return dictionary_generations;
}

TTableGenerations aggregate_two_leaf_table_generations(const TTableGenerations& lhs,
                                                       const TTableGenerations& rhs,
                                                       const Catalog_Namespace::Catalog& cat,
                                                       const int64_t start_rowid) {
  CHECK_EQ(lhs.size(), rhs.size());
  TTableGenerations result;
  for (const auto& lhs_table_generation : lhs) {
    const auto& rhs_table_generation = find_table_generation(rhs, lhs_table_generation.table_id);
    const auto td = cat.getMetadataForTable(lhs_table_generation.table_id);
    CHECK(td);
    TTableGeneration table_generation;
    table_generation.table_id = lhs_table_generation.table_id;
    if (table_is_replicated(td)) {
      table_generation.tuple_count = rhs_table_generation.tuple_count;
      table_generation.start_rowid = 0;
    } else {
      table_generation.tuple_count = rhs_table_generation.tuple_count;
#ifdef HAVE_RENDERING
      table_generation.start_rowid = start_rowid;
#else
      table_generation.start_rowid = lhs_table_generation.start_rowid + lhs_table_generation.tuple_count;
#endif  // HAVE_RENDERING
    }
    result.push_back(table_generation);
  }
  return result;
}

void aggregate_table_generations(std::vector<TPendingQuery>& pending_queries, const Catalog_Namespace::Catalog& cat) {
  CHECK(!pending_queries.empty());
  for (size_t i = 1; i < pending_queries.size(); ++i) {
    pending_queries[i].table_generations =
        aggregate_two_leaf_table_generations(pending_queries[i - 1].table_generations,
                                             pending_queries[i].table_generations,
                                             cat,
                                             i * 100 * 1000 * 1000 * 1000L);
  }
}

void check_leaf_layout_consistency(const std::vector<std::shared_ptr<ResultSet>>& leaf_results) {
  if (leaf_results.empty()) {
    return;
  }
  const auto& leaf_result = leaf_results.front();
  const auto& ref_layout = leaf_result->getQueryMemDesc();
  for (const auto& rs : leaf_results) {
    const auto& layout = rs->getQueryMemDesc();
    CHECK(ref_layout.hash_type == layout.hash_type);
    CHECK_EQ(ref_layout.keyless_hash, layout.keyless_hash);
    CHECK(!layout.interleaved_bins_on_gpu);
    CHECK_EQ(ref_layout.idx_target_as_key, layout.idx_target_as_key);
    CHECK_EQ(ref_layout.init_val, layout.init_val);
    CHECK(ref_layout.group_col_widths == layout.group_col_widths);
    CHECK(ref_layout.agg_col_widths == layout.agg_col_widths);
    CHECK(ref_layout.target_groupby_indices == layout.target_groupby_indices);
    if (layout.hash_type != GroupByColRangeType::MultiCol &&
        layout.hash_type != GroupByColRangeType::OneColGuessedRange &&
        layout.hash_type != GroupByColRangeType::Projection) {
      CHECK_EQ(ref_layout.entry_count, layout.entry_count);
    }
    CHECK_EQ(ref_layout.entry_count_small, layout.entry_count_small);
    CHECK_EQ(ref_layout.min_val, layout.min_val);
    CHECK_EQ(ref_layout.max_val, layout.max_val);
    CHECK_EQ(ref_layout.bucket, layout.bucket);
    CHECK_EQ(ref_layout.has_nulls, layout.has_nulls);
    CHECK(ref_layout.sharing == layout.sharing);
    CHECK_EQ(ref_layout.count_distinct_descriptors_.size(), layout.count_distinct_descriptors_.size());
    // Count distinct descriptors can legitimately differ in device only.
    for (size_t i = 0; i < ref_layout.count_distinct_descriptors_.size(); ++i) {
      auto ref_count_distinct_desc = ref_layout.count_distinct_descriptors_[i];
      auto count_distinct_desc = layout.count_distinct_descriptors_[i];
      count_distinct_desc.device_type = ref_count_distinct_desc.device_type;
      CHECK(ref_count_distinct_desc == count_distinct_desc);
    }
    CHECK(!layout.sort_on_gpu_);
    CHECK(!layout.output_columnar);
    CHECK(ref_layout.key_column_pad_bytes == layout.key_column_pad_bytes);
    CHECK(ref_layout.target_column_pad_bytes == layout.target_column_pad_bytes);
  }
}

void check_replicated_table_consistency(const std::vector<TPendingQuery>& pending_queries,
                                        const Catalog_Namespace::Catalog& cat,
                                        const int table_id) {
  CHECK(!pending_queries.empty());
  const auto td = cat.getMetadataForTable(table_id);
  CHECK(td);
  if (!table_is_replicated(td)) {
    return;
  }
  const auto& ref_table_generation = find_table_generation(pending_queries.front().table_generations, table_id);
  for (const auto& pending_query : pending_queries) {
    const auto& crt_table_generation = find_table_generation(pending_query.table_generations, table_id);
    if (crt_table_generation.tuple_count != ref_table_generation.tuple_count) {
      throw std::runtime_error("Inconsistent size for replicated table " + td->tableName +
                               " across leaves not supported yet");
    }
  }
}

void check_replicated_tables_consistency(const std::vector<TPendingQuery>& pending_queries,
                                         const Catalog_Namespace::Catalog& cat) {
  CHECK(!pending_queries.empty());
  const TPendingQuery& pending_query = pending_queries.front();
  for (const TTableGeneration& table_generation : pending_query.table_generations) {
    check_replicated_table_consistency(pending_queries, cat, table_generation.table_id);
  }
}

}  // namespace

// TODO(alex): split and clean-up this method
AggregatedResult LeafAggregator::execute(const Catalog_Namespace::SessionInfo& parent_session_info,
                                         const std::string& query_ra,
                                         const ExecutionOptions& eo) {
  const auto clock_begin = timer_start();
  std::lock_guard<std::mutex> execution_lock(execution_mutex_);
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto queue_time_ms = timer_stop(clock_begin);
  auto pending_queries = startQueryOnLeaves(parent_session_info, query_ra);
  // Don't support inconsistent replicated table size across leaves, we need to
  // add an API for retrieving tables at a given generation in the fragmenter.
  check_replicated_tables_consistency(pending_queries, parent_session_info.get_catalog());
  aggregate_table_generations(pending_queries, parent_session_info.get_catalog());
  const auto column_ranges = aggregate_leaf_ranges(parent_session_info.get_catalog(), pending_queries);
  const auto string_dictionary_generations = aggregate_dictionary_generations(pending_queries);
  for (auto& pending_query : pending_queries) {
    pending_query.column_ranges = column_ranges;
    pending_query.dictionary_generations = string_dictionary_generations;
  }
  bool execution_finished = false;
  unsigned node_id{0};
  TMergeType::type merge_type = TMergeType::REDUCE;
  const auto& cat = parent_session_info.get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId);
  RelAlgExecutor ra_executor(executor.get(), cat);
  ra_executor.prepareLeafExecution(column_ranges_from_thrift(column_ranges),
                                   string_dictionary_generations_from_thrift(string_dictionary_generations),
                                   {});
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
      check_replication_constraints(ra.get());
      replicated = input_is_replicated(ra.get());
    } else {
      CHECK_GE(crt_subquery_idx, 0);
      const auto subquery_ra = subqueries[crt_subquery_idx]->getRelAlg();
      check_replication_constraints(subquery_ra);
      replicated = input_is_replicated(subquery_ra);
    }
    std::vector<std::shared_ptr<ResultSet>> leaf_results;
    TRowDescriptor row_desc;
    std::vector<TargetInfo> target_infos;
    std::vector<std::future<std::unique_ptr<ResultSet>>> leaf_futures;
    for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
      leaf_futures.emplace_back(std::async(std::launch::async,
                                           [&execution_finished,
                                            &leaves_mutex,
                                            &merge_type,
                                            &node_id,
                                            &pending_queries,
                                            &query_ra,
                                            &ra_executor,
                                            &row_desc,
                                            &target_infos,
                                            leaf_idx,
                                            this] {
                                             TStepResult step_result;
                                             const auto& leaf = leaves_[leaf_idx];
                                             leaf->execute_first_step(step_result, pending_queries[leaf_idx]);
                                             auto result_set = ResultSet::unserialize(step_result.serialized_rows,
                                                                                      ra_executor.getExecutor());
                                             std::lock_guard<std::mutex> lock(leaves_mutex);
                                             target_infos = result_set->getTargetInfos();
                                             if (leaf_idx == 0) {
                                               execution_finished = step_result.execution_finished;
                                               merge_type = step_result.merge_type;
                                               row_desc = step_result.row_desc;
                                               node_id = step_result.node_id;
                                             }
                                             return result_set;
                                           }));
    }
    for (auto& leaf_future : leaf_futures) {
      leaf_future.wait();
    }
    for (size_t leaf_idx = 0; leaf_idx < leaf_futures.size(); ++leaf_idx) {
      auto& leaf_future = leaf_futures[leaf_idx];
      auto result_set = leaf_future.get();
      if (!result_set->definitelyHasNoRows()) {
        leaf_results.emplace_back(result_set.release());
      }
      if (replicated) {
        break;
      }
    }
    QueryMemoryDescriptor empty_query_mem_desc{};
    auto empty_result_set =
        std::make_shared<ResultSet>(target_infos, ExecutorDeviceType::CPU, empty_query_mem_desc, nullptr, nullptr);
    std::shared_ptr<ResultSet> reduced_rs;
    if (merge_type == TMergeType::UNION) {
      if (leaf_results.empty()) {
        reduced_rs = empty_result_set;
      } else {
        reduced_rs = leaf_results.front();
        for (size_t i = 1; i < leaf_results.size(); ++i) {
          reduced_rs->append(*leaf_results[i]);
        }
      }
    } else {
      check_leaf_layout_consistency(leaf_results);
      std::vector<ResultSet*> leaf_results_ptrs;
      for (auto& result : leaf_results) {
        leaf_results_ptrs.push_back(result.get());
      }
      ResultSetManager rs_manager;
      if (leaf_results_ptrs.size() == 1) {
        reduced_rs = leaf_results.front();
      } else if (leaf_results_ptrs.size() > 1) {
        rs_manager.reduce(leaf_results_ptrs);
        auto rs_manager_rs = rs_manager.getOwnResultSet();
        reduced_rs = rs_manager_rs ? rs_manager_rs : leaf_results.front();
      } else {
        reduced_rs = empty_result_set;
      }
    }
    const auto target_meta_infos = target_meta_infos_from_thrift(row_desc);
    AggregatedResult leaves_result{reduced_rs, target_meta_infos};
    CompilationOptions co = {ExecutorDeviceType::CPU, true, ExecutorOptLevel::Default, false};
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
      aggregated_rs->setQueueTime(queue_time_ms);
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
      const auto& aggregated_result_rows = aggregated_result->getRows();
      auto aggregated_rs =
          aggregated_result_rows.definitelyHasNoRows() ? empty_result_set : aggregated_result_rows.getResultSet();
      CHECK(aggregated_rs);
      broadcastResultSet(aggregated_rs.get(), row_desc, pending_queries);
    }
  }
  CHECK(false);
  return {nullptr, {}};
}

std::string LeafAggregator::render(const Catalog_Namespace::SessionInfo& parent_session_info,
                                   const std::string& vega_json,
                                   const int64_t widget_id,
                                   const int compressionLevel) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  auto& leaf_session_ids = session_it->second;
  std::vector<std::future<std::string>> leaf_futures;
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    leaf_futures.emplace_back(
        std::async(std::launch::async, [&leaf_session_ids, &vega_json, leaf_idx, widget_id, compressionLevel, this] {
          TRenderResult render_result;
          leaves_[leaf_idx]->render_vega(
              render_result, leaf_session_ids[leaf_idx], widget_id, vega_json, compressionLevel, "");
          return render_result.image;
        }));
  }
  for (auto& leaf_future : leaf_futures) {
    leaf_future.wait();
  }
  for (auto& leaf_future : leaf_futures) {
    // TODO(alex): composite
    return leaf_future.get();
  }
  CHECK(false);
  return "";
}

TPixelTableRowResult LeafAggregator::getResultRowForPixel(
    const Catalog_Namespace::SessionInfo& parent_session_info,
    const int64_t widget_id,
    const TPixel& pixel,
    const std::map<std::string, std::vector<std::string>>& table_col_names,
    const bool column_format,
    const int32_t pixel_radius) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  auto& leaf_session_ids = session_it->second;
  std::vector<std::future<TPixelTableRowResult>> leaf_futures;
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    leaf_futures.emplace_back(std::async(
        std::launch::async,
        [&leaf_session_ids, &pixel, &table_col_names, column_format, leaf_idx, pixel_radius, widget_id, this] {
          TPixelTableRowResult lookup_result;
          leaves_[leaf_idx]->get_result_row_for_pixel(lookup_result,
                                                      leaf_session_ids[leaf_idx],
                                                      widget_id,
                                                      pixel,
                                                      table_col_names,
                                                      column_format,
                                                      pixel_radius,
                                                      "");
          return lookup_result;
        }));
  }
  for (auto& leaf_future : leaf_futures) {
    leaf_future.wait();
  }
  for (auto& leaf_future : leaf_futures) {
    // TODO(alex): composite
    return leaf_future.get();
  }
  CHECK(false);
  return {};
}

std::vector<TQueryResult> LeafAggregator::forwardQueryToLeaves(
    const Catalog_Namespace::SessionInfo& parent_session_info,
    const std::string& query_str) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  auto& leaf_session_ids = session_it->second;
  std::vector<std::future<TQueryResult>> leaf_futures;
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    leaf_futures.emplace_back(std::async(std::launch::async, [&leaf_session_ids, &query_str, leaf_idx, this] {
      TQueryResult query_result;
      leaves_[leaf_idx]->sql_execute(query_result, leaf_session_ids[leaf_idx], query_str, true, "");
      return query_result;
    }));
  }
  for (auto& leaf_future : leaf_futures) {
    leaf_future.wait();
  }
  std::vector<TQueryResult> results;
  for (auto& leaf_future : leaf_futures) {
    results.push_back(leaf_future.get());
  }
  return results;
}

TQueryResult LeafAggregator::forwardQueryToLeaf(const Catalog_Namespace::SessionInfo& parent_session_info,
                                                const std::string& query_str,
                                                const size_t leaf_idx) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  auto& leaf_session_ids = session_it->second;
  CHECK_LT(leaf_idx, leaves_.size());
  TQueryResult query_result;
  leaves_[leaf_idx]->sql_execute(query_result, leaf_session_ids[leaf_idx], query_str, true, "");
  return query_result;
}

std::vector<TPendingQuery> LeafAggregator::startQueryOnLeaves(const Catalog_Namespace::SessionInfo& parent_session_info,
                                                              const std::string& query_ra) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(parent_session_info.get_session_id());
  auto& leaf_session_ids = session_it->second;
  CHECK_EQ(leaves_.size(), leaf_session_ids.size());
  std::vector<std::future<TPendingQuery>> leaf_futures;
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    leaf_futures.emplace_back(
        std::async(std::launch::async, [leaf_idx, &leaf_session_ids, &parent_session_info, &query_ra, this] {
          TPendingQuery pending_query;
          try {
            leaves_[leaf_idx]->start_query(pending_query, leaf_session_ids[leaf_idx], query_ra);
          } catch (const TException&) {
            const auto credentials_it = session_credentials_.find(parent_session_info.get_session_id());
            CHECK(credentials_it != session_credentials_.end());
            const auto& credentials = credentials_it->second;
            leaf_session_ids[leaf_idx] =
                leaves_[leaf_idx]->connect(credentials.user, credentials.passwd, credentials.dbname);
            leaves_[leaf_idx]->start_query(pending_query, leaf_session_ids[leaf_idx], query_ra);
          }
          return pending_query;
        }));
  }
  for (auto& leaf_future : leaf_futures) {
    leaf_future.wait();
  }
  std::vector<TPendingQuery> pending_queries;
  for (auto& leaf_future : leaf_futures) {
    pending_queries.push_back(leaf_future.get());
  }
  return pending_queries;
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
  {
    const auto it_ok = leaf_sessions_.emplace(parent_session_info.get_session_id(), leaf_session_ids);
    CHECK(it_ok.second);
  }
  {
    const auto it_ok =
        session_credentials_.emplace(parent_session_info.get_session_id(), Credentials{user, passwd, dbname});
    CHECK(it_ok.second);
  }
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
  session_credentials_.erase(session);
}

void LeafAggregator::interrupt(const TSessionId session) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(leaf_sessions_mutex_);
  const auto session_it = getSessionIterator(session);
  const auto& leaf_session_ids = session_it->second;
  CHECK_EQ(leaves_.size(), leaf_session_ids.size());
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    const auto& leaf = leaves_[leaf_idx];
    leaf->interrupt(leaf_session_ids[leaf_idx]);
  }
}

size_t LeafAggregator::leafCount() const {
  return leaves_.size();
}

void LeafAggregator::broadcastResultSet(const ResultSet* result_set,
                                        const TRowDescriptor& row_desc,
                                        const std::vector<TPendingQuery>& pending_queries) const {
  CHECK_EQ(leaves_.size(), pending_queries.size());
  const auto serialized_result_set = result_set->serialize();
  std::vector<std::future<void>> leaf_futures;
  for (size_t leaf_idx = 0; leaf_idx < leaves_.size(); ++leaf_idx) {
    leaf_futures.emplace_back(
        std::async(std::launch::async, [&pending_queries, &row_desc, &serialized_result_set, leaf_idx, this] {
          const auto& leaf = leaves_[leaf_idx];
          const auto query_id = pending_queries[leaf_idx].id;
          leaf->broadcast_serialized_rows(serialized_result_set, row_desc, query_id);
        }));
  }
  for (auto& leaf_future : leaf_futures) {
    leaf_future.wait();
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
