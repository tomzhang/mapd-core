#ifndef QUERYRENDERER_ROOTCACHE_H_
#define QUERYRENDERER_ROOTCACHE_H_

#include "PerGpuData.h"
#include "Data/QueryPolyDataTable.h"
#include "Utils/Utils.h"
#include "Rendering/QueryBufferPool.h"
#include <QueryEngine/ResultRows.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/sequenced_index.hpp>

namespace QueryRenderer {

class SqlPolyQueryCacheMap {
 public:
  SqlPolyQueryCacheMap(RootCache* prntGpuCache, const size_t maxCacheBytes);
  ~SqlPolyQueryCacheMap() {}

  GpuId getGpuIdForPolyTableCache(const std::string& polyTableName);
  void updateCache(const RootCacheShPtr& qrmGpuCache,
                   const GpuId gpuId,
                   const std::string& polyTableName,
                   const std::string& queryStr,
                   const PolyTableByteData& initTableData,
                   const QueryDataLayoutShPtr& vertLayoutPtr = nullptr,
                   const PolyRowDataShPtr& rowDataPtr = nullptr,
                   const QueryDataLayoutShPtr& uniformLayoutPtr = nullptr);

  void deleteCache(const std::string& polyTableName, const std::string& queryStr = "");
  void clear() {
    _polyCaches.clear();
    _polyTableMap.clear();
  }
  PolyTableByteData getPolyBufferByteData(GpuId gpuId, const std::string& polyTableName, const std::string& queryStr);
  PolyTableDataInfo getPolyBufferData(GpuId gpuId, const std::string& polyTableName, const std::string& queryStr);
  PolyCudaHandles getCudaHandlesPreQuery(const RootCacheShPtr& qrmGpuCache,
                                         GpuId gpuId,
                                         const std::string& polyTableName,
                                         const std::string& queryStr);
  void updateCachePostQuery(GpuId gpuId,
                            const std::string& polyTableName,
                            const std::string& queryStr,
                            const QueryDataLayoutShPtr& uniformLayoutPtr,
                            const QueryDataLayoutShPtr& vertLayoutPtr = nullptr,
                            const PolyRowDataShPtr& rowDataPtr = nullptr);

  SqlQueryPolyDataTableCacheShPtr getPolyTableCache(const std::string& polyTableName, const std::string& queryStr);
  void notifyCacheRelease(const SqlQueryPolyDataTableCacheShPtr& releasedCachePtr,
                          const std::string* swapQueryStr = nullptr);

  bool hasCache(const std::string& tableName) const;
  bool hasCache(const std::string& tableName, const GpuId gpuId) const;
  bool hasCache(const std::string& tableName, const std::string& queryStr) const;
  bool hasCache(const std::string& tableName, const std::string& queryStr, const GpuId gpuId) const;

  void clearOverflowBuffers();

 private:
  static size_t buildSqlHash(const std::string& queryStr) { return std::hash<std::string>{}(queryStr); }

  struct PolySqlCache {
    SqlQueryPolyDataTableCacheShPtr sqlPolyDataTablePtr;
    size_t totalUsedBytes;
    std::chrono::milliseconds lastUsedTime;

    PolySqlCache(const SqlQueryPolyDataTableCacheShPtr& sqlPolyDataTablePtr)
        : sqlPolyDataTablePtr(sqlPolyDataTablePtr), totalUsedBytes(0) {
      lastUsedTime = getCurrentTimeMS();
    }

    std::string getSqlQueryStr() const { return sqlPolyDataTablePtr->getSqlQueryStrRef(); }
  };

  struct ChangeLastUsedTime {
    ChangeLastUsedTime() : new_time(getCurrentTimeMS()) {}
    void operator()(PolySqlCache& cache) { cache.lastUsedTime = new_time; }

   private:
    std::chrono::milliseconds new_time;
  };

  struct ChangeTotalUsedBytes {
    ChangeTotalUsedBytes(size_t byteDiff) : byteDiff(byteDiff) {}
    void operator()(PolySqlCache& cache) { cache.totalUsedBytes += byteDiff; }

   private:
    int64_t byteDiff;
  };

  struct CachePtrTag {};
  struct CacheSqlTag {};
  struct CacheLastUsedTimeTag {};

  typedef ::boost::multi_index_container<
      PolySqlCache,
      ::boost::multi_index::indexed_by<
          ::boost::multi_index::hashed_unique<::boost::multi_index::tag<CachePtrTag>,
                                              ::boost::multi_index::member<PolySqlCache,
                                                                           SqlQueryPolyDataTableCacheShPtr,
                                                                           &PolySqlCache::sqlPolyDataTablePtr>>,

          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::tag<CacheSqlTag>,
              ::boost::multi_index::const_mem_fun<PolySqlCache, std::string, &PolySqlCache::getSqlQueryStr>>,

          ::boost::multi_index::ordered_non_unique<
              ::boost::multi_index::tag<CacheLastUsedTimeTag>,
              ::boost::multi_index::member<PolySqlCache, std::chrono::milliseconds, &PolySqlCache::lastUsedTime>>>>
      PolySqlCacheMap;

  typedef PolySqlCacheMap::index<CachePtrTag>::type SqlCache_by_ptr;
  typedef PolySqlCacheMap::index<CacheSqlTag>::type SqlCache_by_sql;
  typedef PolySqlCacheMap::index<CacheLastUsedTimeTag>::type SqlCache_by_lastTime;

  struct PerGpuData;

  struct PolyVertCache {
    QueryVertexBufferShPtr vbo;
    QueryIndexBufferShPtr ibo;
    PolyRowDataShPtr rowDataPtr;
    PolySqlCacheMap sqlCacheMap;
    PerGpuData* perGpuDataPtr;
    std::string polyTableName;
    size_t totalUsedBytes;
    bool isOverflow;

    explicit PolyVertCache(PerGpuData* perGpuDataPtr, const std::string& polyTableName, const bool isOverflow)
        : perGpuDataPtr(perGpuDataPtr), polyTableName(polyTableName), totalUsedBytes(0), isOverflow(isOverflow) {}

    std::chrono::milliseconds getLastUpdateTime() const {
      if (sqlCacheMap.size()) {
        auto& sqlLookup = sqlCacheMap.get<CacheLastUsedTimeTag>();
        return sqlLookup.rbegin()->lastUsedTime;
      }
      return std::chrono::milliseconds(0);
    }
  };

  struct CachePolyTag {};
  struct CachePolyPtrTag {};
  typedef ::boost::multi_index_container<
      std::shared_ptr<PolyVertCache>,
      ::boost::multi_index::indexed_by<
          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::tag<CachePolyTag>,
              ::boost::multi_index::member<PolyVertCache, std::string, &PolyVertCache::polyTableName>>,
          ::boost::multi_index::hashed_unique<::boost::multi_index::tag<CachePolyPtrTag>,
                                              ::boost::multi_index::identity<std::shared_ptr<PolyVertCache>>>>>
      PolyCacheMap;

  typedef PolyCacheMap::index<CachePolyTag>::type PolyCache_by_tableName;
  typedef PolyCacheMap::index<CachePolyPtrTag>::type PolyCache_by_ptr;

  struct PerGpuData : BasePerGpuData {
    PolyCacheMap polyCacheMap;
    PolyCacheMap overflowPolyMap;
    std::shared_ptr<QueryBufferPool<QueryResultVertexBuffer>> vboPool;
    std::shared_ptr<QueryBufferPool<QueryResultIndexBuffer>> iboPool;
    std::shared_ptr<QueryBufferPool<QueryResultUniformBuffer>> uboPool;
    std::shared_ptr<QueryBufferPool<QueryResultIndirectIbo>> indiboPool;
    std::shared_ptr<QueryBufferPool<QueryResultIndirectVbo>> indvboPool;
    size_t totalUsedBytes;

    explicit PerGpuData(const RootPerGpuDataShPtr& rootData) : BasePerGpuData(rootData), totalUsedBytes(0) {
      auto renderer = getGLRendererShPtr();
      vboPool.reset(new QueryBufferPool<QueryResultVertexBuffer>(renderer));
      iboPool.reset(new QueryBufferPool<QueryResultIndexBuffer>(renderer));
      uboPool.reset(new QueryBufferPool<QueryResultUniformBuffer>(renderer));
      indiboPool.reset(new QueryBufferPool<QueryResultIndirectIbo>(renderer));
      indvboPool.reset(new QueryBufferPool<QueryResultIndirectVbo>(renderer));
    }

    ~PerGpuData() {
      // need to make active to properly delete gpu resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }
  };

  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  PerGpuDataMap _polyCaches;
  PolyCacheMap _polyTableMap;

  RootCache* prntGpuCache;
  const size_t _maxCacheBytes;

  static const std::chrono::milliseconds maxRsrcIdleTime;
  static size_t _inactivatePolyDataTableCaches(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                               BaseQueryPolyDataTable::PerGpuData* cacheBuffers,
                                               const bool deleteRsrc);

  static size_t _inactivatePolyDataTable(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                         SqlQueryPolyDataTableCacheShPtr& sqlPolyDataTablePtr,
                                         const bool deleteRsrc);

  static size_t _inactivatePolyVertCache(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                         std::shared_ptr<PolyVertCache>& polyVertCache,
                                         const bool deleteRsrc);

  static bool _cleanOldCachesToMakeRoom(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                        PolyCacheMap* polyTableMap,
                                        const size_t maxNumBytes,
                                        const size_t requiredNumBytes,
                                        const std::chrono::milliseconds& cutoffTime,
                                        const std::string* polyTableNameToKeep = nullptr,
                                        const bool force = false);

  static void _clearOverflowBuffers(SqlPolyQueryCacheMap::PerGpuData* gpuData, PolyCacheMap& polyTableMap);

  static std::tuple<bool, PolyCacheMap::iterator, SqlCache_by_sql::iterator>
  _getSqlItr(const PolyCacheMap& polyMap, const std::string& polyTableName, const std::string& queryStr);

  static std::tuple<bool, PolyCacheMap::iterator, SqlCache_by_sql::iterator>
  _getSqlItr(SqlPolyQueryCacheMap::PerGpuData* gpuData, const std::string& polyTableName, const std::string& queryStr);
};

struct NonProjectionRenderQueryCache {
  TableId tableId;
  std::string sqlStr;
  size_t cachedBytes;
  std::shared_ptr<ResultRows> results;
  std::vector<TargetMetaInfo> resultsRowShape;
  std::vector<std::pair<TableId, std::string>> usedTablesInQuery;
  std::chrono::milliseconds lastUsedTime;

  explicit NonProjectionRenderQueryCache(const TableId tableId,
                                         const std::string& sqlStr,
                                         const size_t numBytes,
                                         const std::shared_ptr<ResultRows>& results,
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
                                           const std::shared_ptr<ResultRows>& results,
                                           const std::vector<TargetMetaInfo>&& resultRowShape,
                                           const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery);

  NPRQueryCacheShPtr addQueryOnlyToCache(const std::string& sqlStr,
                                         const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery);

  NPRQueryCacheShPtr updateQueryResultsInCache(const std::string& sqlStr,
                                               const std::shared_ptr<ResultRows>& results,
                                               const std::vector<TargetMetaInfo>&& resultRowShape);
  NPRQueryCacheShPtr updateQueryResultsInCache(const TableId tableId,
                                               const std::shared_ptr<ResultRows>& results,
                                               const std::vector<TargetMetaInfo>&& resultRowShape);

  NPRQueryCacheShPtr removeQueryResultsFromCache(const TableId tableId);
  NPRQueryCacheShPtr removeQueryResultsFromCache(const std::string& sqlStr);

  bool hasQueryCache(const std::string& sqlStr);
  bool hasQueryCache(const TableId tableId);

  std::string getQueryForCache(const TableId tableId) const;
  std::pair<TableId, std::string> getQueryCachePrimaryTableInfo(const TableId tableId) const;

  bool canFitResults(const std::shared_ptr<ResultRows>& newResults);
  bool canFitUpdatedResults(const std::string& sqlStr, const std::shared_ptr<ResultRows>& updatedResults);
  bool canFitUpdatedResults(const TableId tableId, const std::shared_ptr<ResultRows>& updatedResults);

  std::pair<const ResultRows*, const std::vector<TargetMetaInfo>*> getQueryCacheResults(const TableId tableId) const;

 private:
  static const std::chrono::milliseconds maxCacheIdleTime;

  size_t _totalCachedBytes;
  size_t _maxTotalCachedBytes;

  struct TableIdTag {};
  struct SqlStrTag {};

  struct UpdateCacheResults {
    UpdateCacheResults(const std::shared_ptr<ResultRows>& results,
                       const std::vector<TargetMetaInfo>&& resultsRowShape,
                       const size_t numBytes,
                       bool udpateTime = true);

    void operator()(NPRQueryCacheShPtr& renderCache);

   private:
    std::chrono::milliseconds new_time;
    std::shared_ptr<ResultRows> new_results;
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

  std::shared_ptr<SqlPolyQueryCacheMap> polyCacheMap;

  size_t numSamples;
  NonProjectionRenderQueryCacheMap renderQueryCacheMap;

  RootCache(const size_t numSamples, const size_t maxCacheSize = 1000000)
      : perGpuData(new RootPerGpuDataMap()), numSamples(numSamples), renderQueryCacheMap(maxCacheSize) {}

  bool hasPolyTableGpuCache(const std::string& tableName) { return polyCacheMap->hasCache(tableName); }
  bool hasPolyTableGpuCache(const std::string& tableName, const GpuId gpuId) {
    return polyCacheMap->hasCache(tableName, gpuId);
  }

  bool hasPolyTableGpuCache(const std::string& tableName, const std::string& sqlStr) {
    return polyCacheMap->hasCache(tableName, sqlStr);
  }

  bool hasPolyTableGpuCache(const std::string& tableName, const std::string& sqlStr, const GpuId gpuId) {
    return polyCacheMap->hasCache(tableName, sqlStr, gpuId);
  }

  bool supportsInt64() const;
  DistributedRenderBufferCompositorShPtr getDistributedCompositorPtr();

 private:
  mutable std::shared_ptr<bool> _supportsInt64Ptr;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_ROOTCACHE_H_
