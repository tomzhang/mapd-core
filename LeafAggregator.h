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

class PresistentLeafClient {
 public:
  PresistentLeafClient(const LeafHostInfo& leaf_host);

  TSessionId connect(const std::string& user, const std::string& passwd, const std::string& dbname);
  void disconnect(const TSessionId session);
  void start_query(TPendingQuery& _return, const TSessionId session, const std::string& query_ra);
  void execute_first_step(TStepResult& _return, const TPendingQuery& pending_query);
  void broadcast_serialized_rows(const std::string& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id);

 private:
  void setupClient();

  const LeafHostInfo leaf_host_;
  std::unique_ptr<MapDClient> client_;
};

class LeafAggregator {
 public:
  LeafAggregator(const std::vector<LeafHostInfo>& leaves);

  AggregatedResult execute(const Catalog_Namespace::SessionInfo& parent_session_info, const std::string& query_ra);

  void connect(const Catalog_Namespace::SessionInfo& parent_session_info,
               const std::string& user,
               const std::string& passwd,
               const std::string& dbname);

  void disconnect(const TSessionId session);

  size_t leafCount() const;

 private:
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

  std::vector<std::unique_ptr<PresistentLeafClient>> leaves_;
  SessionMap leaf_sessions_;  // map from aggregator session to leaf sessions
  std::unordered_map<TSessionId, Credentials> session_credentials_;
  mapd_shared_mutex leaf_sessions_mutex_;
};

#endif  // LEAFAGGREGATOR_H
