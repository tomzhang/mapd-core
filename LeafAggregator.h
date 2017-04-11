#ifndef LEAFAGGREGATOR_H
#define LEAFAGGREGATOR_H

#include "LeafHostInfo.h"
#include "QueryEngine/TargetMetaInfo.h"
#ifdef HAVE_RENDERING
#include "QueryRenderer/QueryRenderManager.h"
#endif  // HAVE_RENDERING
#include "Shared/mapd_shared_mutex.h"
#include "gen-cpp/MapD.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

class MapDClient;
class ResultSet;

namespace Catalog_Namespace {
class SessionInfo;
}  // Catalog_Namespace

struct AggregatedResult {
  std::shared_ptr<ResultSet> rs;
  const std::vector<TargetMetaInfo> targets_meta;
};

class PersistentLeafClient {
 public:
  PersistentLeafClient(const LeafHostInfo& leaf_host, const bool with_timeout) noexcept;

  TSessionId connect(const std::string& user, const std::string& passwd, const std::string& dbname);
  void disconnect(const TSessionId session);
  void interrupt(const TSessionId session);
  void start_query(TPendingQuery& _return,
                   const TSessionId session,
                   const std::string& query_ra,
                   const bool just_explain);
  void execute_first_step(TStepResult& _return, const TPendingQuery& pending_query);
  void broadcast_serialized_rows(const std::string& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id);
  void insert_data(const TSessionId session, const TInsertData& thrift_insert_data);
  void checkpoint(const TSessionId session, const int32_t db_id, const int32_t table_id);
  void render_vega_raw_pixels(TRawPixelDataResult& _return,
                              const TSessionId session,
                              const int64_t widget_id,
                              const std::string& vega_json);
  void get_result_row_for_pixel(TPixelTableRowResult& _return,
                                const TSessionId session,
                                const int64_t widget_id,
                                const TPixel& pixel,
                                const std::map<std::string, std::vector<std::string>>& table_col_names,
                                const bool column_format,
                                const int32_t pixel_radius,
                                const std::string& nonce);
  void sql_execute(TQueryResult& _return,
                   const TSessionId session,
                   const std::string& query_str,
                   const bool column_format,
                   const std::string& nonce);
  void set_execution_mode(const TSessionId session, const TExecuteMode::type mode);

 private:
  void setupClientIfNull();
  void setupClient();

  const LeafHostInfo leaf_host_;
  const bool with_timeout_;
  std::unique_ptr<MapDClient> client_;
  std::mutex client_mutex_;
};

struct ExecutionOptions;

class LeafAggregator {
 public:
  LeafAggregator(const std::vector<LeafHostInfo>& leaves);

  AggregatedResult execute(const Catalog_Namespace::SessionInfo& parent_session_info,
                           const std::string& query_ra,
                           const ExecutionOptions& eo);

#ifdef HAVE_RENDERING
  std::string render(const Catalog_Namespace::SessionInfo& parent_session_info,
                     const std::string& vega_json,
                     const int64_t widget_id,
                     const int compressionLevel,
                     QueryRenderer::QueryRenderManager* render_manager);
#endif  // HAVE_RENDERING

  TPixelTableRowResult getResultRowForPixel(const Catalog_Namespace::SessionInfo& parent_session_info,
                                            const int64_t widget_id,
                                            const TPixel& pixel,
                                            const std::map<std::string, std::vector<std::string>>& table_col_names,
                                            const bool column_format,
                                            const int32_t pixelRadius);

  // Used for DDL statements
  std::vector<TQueryResult> forwardQueryToLeaves(const Catalog_Namespace::SessionInfo& parent_session_info,
                                                 const std::string& query_str);

  TQueryResult forwardQueryToLeaf(const Catalog_Namespace::SessionInfo& parent_session_info,
                                  const std::string& query_str,
                                  const size_t leaf_idx);

  void insertDataToLeaf(const Catalog_Namespace::SessionInfo& parent_session_info,
                        const size_t leaf_idx,
                        const TInsertData& thrift_insert_data);

  void checkpointLeaf(const Catalog_Namespace::SessionInfo& parent_session_info,
                      const size_t leaf_idx,
                      const int32_t db_id,
                      const int32_t table_id);

  void connect(const Catalog_Namespace::SessionInfo& parent_session_info,
               const std::string& user,
               const std::string& passwd,
               const std::string& dbname);

  void disconnect(const TSessionId session);

  void interrupt(const TSessionId session);

  void set_execution_mode(const TSessionId session, const TExecuteMode::type mode);

  size_t leafCount() const;

 private:
  std::vector<TPendingQuery> startQueryOnLeaves(const Catalog_Namespace::SessionInfo& parent_session_info,
                                                const std::string& query_ra,
                                                const bool just_explain);

  void broadcastResultSet(const ResultSet* result_set,
                          const TRowDescriptor& row_desc,
                          const std::vector<TPendingQuery>& pending_queries) const;

  typedef std::map<TSessionId, std::vector<TSessionId>> SessionMap;

  SessionMap::iterator getSessionIterator(const TSessionId session);

  struct Credentials {
    const std::string user;
    const std::string passwd;
    const std::string dbname;
  };

  std::vector<std::unique_ptr<PersistentLeafClient>> leaves_;
  std::vector<std::unique_ptr<PersistentLeafClient>> leaves_no_timeout_;
  SessionMap leaf_sessions_;  // map from aggregator session to leaf sessions
  std::unordered_map<TSessionId, Credentials> session_credentials_;
  mapd_shared_mutex leaf_sessions_mutex_;
  std::mutex execution_mutex_;
};

#endif  // LEAFAGGREGATOR_H
