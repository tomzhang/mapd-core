#ifndef LEAFAGGREGATOR_H
#define LEAFAGGREGATOR_H

#include "LeafHostInfo.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "Shared/mapd_shared_mutex.h"
#include "gen-cpp/mapd_types.h"

#include <cstdint>
#include <memory>
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
  PersistentLeafClient(const LeafHostInfo& leaf_host) noexcept;

  TSessionId connect(const std::string& user, const std::string& passwd, const std::string& dbname);
  void disconnect(const TSessionId session);
  void interrupt(const TSessionId session);
  void start_query(TPendingQuery& _return, const TSessionId session, const std::string& query_ra);
  void execute_first_step(TStepResult& _return, const TPendingQuery& pending_query);
  void broadcast_serialized_rows(const std::string& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id);
  void render_vega(TRenderResult& _return,
                   const TSessionId session,
                   const int64_t widget_id,
                   const std::string& vega_json,
                   const int compressionLevel,
                   const std::string& nonce);
  void get_result_row_for_pixel(TPixelTableRowResult& _return,
                                const TSessionId session,
                                const int64_t widget_id,
                                const TPixel& pixel,
                                const std::map<std::string, std::vector<std::string>>& table_col_names,
                                const bool column_format,
                                const int32_t pixel_radius,
                                const std::string& nonce);

 private:
  void setupClientIfNull();
  void setupClient();

  const LeafHostInfo leaf_host_;
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

  std::string render(const Catalog_Namespace::SessionInfo& parent_session_info,
                     const std::string& vega_json,
                     const int64_t widget_id,
                     const int compressionLevel);

  TPixelTableRowResult getResultRowForPixel(const Catalog_Namespace::SessionInfo& parent_session_info,
                                            const int64_t widget_id,
                                            const TPixel& pixel,
                                            const std::map<std::string, std::vector<std::string>>& table_col_names,
                                            const bool column_format,
                                            const int32_t pixelRadius);

  void connect(const Catalog_Namespace::SessionInfo& parent_session_info,
               const std::string& user,
               const std::string& passwd,
               const std::string& dbname);

  void disconnect(const TSessionId session);

  void interrupt(const TSessionId session);

  size_t leafCount() const;

 private:
  std::vector<TPendingQuery> startQueryOnLeaves(const Catalog_Namespace::SessionInfo& parent_session_info,
                                                const std::string& query_ra);

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
  SessionMap leaf_sessions_;  // map from aggregator session to leaf sessions
  std::unordered_map<TSessionId, Credentials> session_credentials_;
  mapd_shared_mutex leaf_sessions_mutex_;
  std::mutex execution_mutex_;
};

#endif  // LEAFAGGREGATOR_H
