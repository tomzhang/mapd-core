#ifndef LEAFAGGREGATOR_H
#define LEAFAGGREGATOR_H

#include "LeafHostInfo.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "Shared/mapd_shared_mutex.h"
#include "gen-cpp/mapd_types.h"

#include <cstdint>
#include <memory>
#include <string>

class MapDClient;
class ResultSet;

namespace Catalog_Namespace {
class SessionInfo;
}  // Catalog_Namespace

struct AggregatedResult {
  std::shared_ptr<ResultSet> rs;
  const std::vector<TargetMetaInfo> targets_meta;
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

  std::vector<std::unique_ptr<MapDClient>> leaves_;
  SessionMap leaf_sessions_;  // map from aggregator session to leaf sessions
  mapd_shared_mutex leaf_sessions_mutex_;
};

#endif  // LEAFAGGREGATOR_H
