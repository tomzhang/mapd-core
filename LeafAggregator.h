#ifndef LEAFAGGREGATOR_H
#define LEAFAGGREGATOR_H

#include "LeafHostInfo.h"

#include "gen-cpp/mapd_types.h"

#include <cstdint>
#include <memory>
#include <string>

class MapDClient;
class ResultSet;

namespace Catalog_Namespace {
class SessionInfo;
}  // Catalog_Namespace

class LeafAggregator {
 public:
  LeafAggregator(const std::vector<LeafHostInfo>& leaves);

  void execute(TQueryResult& _return,
               const Catalog_Namespace::SessionInfo& session_info,
               const std::string& query_str,
               const bool column_format,
               const std::string& nonce);

 private:
  void broadcastResultSet(const ResultSet*) const;

  std::vector<std::unique_ptr<MapDClient>> leaves_;
  std::vector<int64_t> pending_queries_;  // pending query id (per leaf), 0 for no pending query
};

#endif  // LEAFAGGREGATOR_H
