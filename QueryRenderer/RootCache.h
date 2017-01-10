#ifndef QUERYRENDERER_ROOTCACHE_H_
#define QUERYRENDERER_ROOTCACHE_H_

#include "PerGpuData.h"
#include "Data/QueryPolyDataTable.h"
#include "Utils/Utils.h"
#include <QueryEngine/ResultRows.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/sequenced_index.hpp>

namespace QueryRenderer {

struct NonProjectionRenderQueryCache {
  TableId tableId;
  std::string sqlStr;
  size_t cachedBytes;
  ResultRows results;
  std::vector<TargetMetaInfo> resultsRowShape;
  std::vector<std::pair<TableId, std::string>> usedTablesInQuery;
  std::chrono::milliseconds lastUsedTime;

  explicit NonProjectionRenderQueryCache(const TableId tableId,
                                         const std::string& sqlStr,
                                         const size_t numBytes,
                                         const ResultRows&& results,
                                         const std::vector<TargetMetaInfo>&& resultsRowShape,
                                         const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery)
      : tableId(tableId),
        sqlStr(sqlStr),
        cachedBytes(numBytes),
        results(results),
        resultsRowShape(resultsRowShape),
        usedTablesInQuery(usedTablesInQuery),
        lastUsedTime(getCurrentTimeMS()) {}

  std::pair<TableId, std::string> getPrimaryTableInfo() const {
    RUNTIME_EX_ASSERT(usedTablesInQuery.size() > 0,
                      "Cannot get primary table info for query \"" + sqlStr + "\". There is no table info supplied.");
    return usedTablesInQuery[0];
  }
};

class NonProjectionRenderQueryCacheMap {
 public:
  static const TableId emptyTableId;
  static const uint16_t maxQueryTime;  // non-projection queries that take longer than
                                       // maxQueryTime can be cached (given that there's
                                       // enough space)
  explicit NonProjectionRenderQueryCacheMap(const size_t maxTotalCachedBytes)
      : _totalCachedBytes(0), _maxTotalCachedBytes(maxTotalCachedBytes) {}

  NPRQueryCacheShPtr addQueryResultToCache(const std::string& sqlStr,
                                           const ResultRows&& results,
                                           const std::vector<TargetMetaInfo>&& resultRowShape,
                                           const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery);

  NPRQueryCacheShPtr addQueryOnlyToCache(const std::string& sqlStr,
                                         const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery);

  NPRQueryCacheShPtr updateQueryResultsInCache(const std::string& sqlStr,
                                               const ResultRows&& results,
                                               const std::vector<TargetMetaInfo>&& resultRowShape);
  NPRQueryCacheShPtr updateQueryResultsInCache(const TableId tableId,
                                               const ResultRows&& results,
                                               const std::vector<TargetMetaInfo>&& resultRowShape);

  NPRQueryCacheShPtr removeQueryResultsFromCache(const TableId tableId);
  NPRQueryCacheShPtr removeQueryResultsFromCache(const std::string& sqlStr);

  bool hasQueryCache(const std::string& sqlStr);
  bool hasQueryCache(const TableId tableId);

  std::string getQueryForCache(const TableId tableId) const;
  std::pair<TableId, std::string> getQueryCachePrimaryTableInfo(const TableId tableId) const;

  bool canFitResults(const ResultRows& newResults);
  bool canFitUpdatedResults(const std::string& sqlStr, const ResultRows& updatedResults);
  bool canFitUpdatedResults(const TableId tableId, const ResultRows& updatedResults);

  std::pair<const ResultRows*, const std::vector<TargetMetaInfo>*> getQueryCacheResults(const TableId tableId) const;

 private:
  static const std::chrono::milliseconds maxCacheIdleTime;

  size_t _totalCachedBytes;
  size_t _maxTotalCachedBytes;

  struct TableIdTag {};
  struct SqlStrTag {};

  struct UpdateCacheResults {
    UpdateCacheResults(const ResultRows&& results,
                       const std::vector<TargetMetaInfo>&& resultsRowShape,
                       const size_t numBytes,
                       bool udpateTime = true);

    void operator()(NPRQueryCacheShPtr& renderCache);

   private:
    std::chrono::milliseconds new_time;
    ResultRows new_results;
    std::vector<TargetMetaInfo> new_results_row_shape;
    size_t new_size;
    bool update_time;
  };

  typedef ::boost::multi_index_container<
      NPRQueryCacheShPtr,
      ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<::boost::multi_index::tag<TableIdTag>,
                                               ::boost::multi_index::member<NonProjectionRenderQueryCache,
                                                                            TableId,
                                                                            &NonProjectionRenderQueryCache::tableId>>,
          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::tag<SqlStrTag>,
              ::boost::multi_index::member<NonProjectionRenderQueryCache,
                                           decltype(NonProjectionRenderQueryCache::sqlStr),
                                           &NonProjectionRenderQueryCache::sqlStr>>>>
      QueryResultMap;

  typedef QueryResultMap::index<SqlStrTag>::type QueryResultMap_by_SqlStr;

  QueryResultMap _queryResultMap;

  TableId _getNextUnusedTableId();
  void _purgeUnusedCaches();
};

struct RootCache {
  std::shared_ptr<RootPerGpuDataMap> perGpuData;
  std::set<std::string> supportedExtensions;

  std::unordered_map<std::string, std::pair<size_t, SqlQueryPolyDataTableCache>> polyCacheMap;
  size_t numSamples;
  NonProjectionRenderQueryCacheMap renderQueryCacheMap;

  RootCache(const size_t numSamples, const size_t maxCacheSize = 1000000)
      : perGpuData(new RootPerGpuDataMap()),
        polyCacheMap(),
        numSamples(numSamples),
        renderQueryCacheMap(maxCacheSize) {}

  size_t buildSqlHash(const std::string& sqlStr) const { return std::hash<std::string>{}(sqlStr); }

  bool hasPolyTableGpuCache(const std::string& tableName);
  bool hasPolyTableGpuCache(const std::string& tableName, const GpuId gpuId);
  bool hasPolyTableGpuCache(const std::string& tableName, const std::string& sqlStr);
  bool hasPolyTableGpuCache(const std::string& tableName, const std::string& sqlStr, const GpuId gpuId);

  bool supportsInt64() const;

 private:
  mutable std::shared_ptr<bool> _supportsInt64Ptr;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_ROOTCACHE_H_
