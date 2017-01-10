#include "RootCache.h"
#include <Rendering/RenderError.h>

namespace QueryRenderer {

const uint16_t NonProjectionRenderQueryCacheMap::maxQueryTime = 250;  // in ms

const std::chrono::milliseconds NonProjectionRenderQueryCacheMap::maxCacheIdleTime =
    std::chrono::milliseconds(300000);  // 5 minutes, in ms
const TableId NonProjectionRenderQueryCacheMap::emptyTableId = -1;

NonProjectionRenderQueryCacheMap::UpdateCacheResults::UpdateCacheResults(
    const ResultRows&& results,
    const std::vector<TargetMetaInfo>&& resultsRowShape,
    const size_t numBytes,
    bool updateTime)
    : new_results(results), new_results_row_shape(resultsRowShape), new_size(numBytes), update_time(updateTime) {
  new_time = getCurrentTimeMS();
}

void NonProjectionRenderQueryCacheMap::UpdateCacheResults::operator()(NPRQueryCacheShPtr& renderCache) {
  if (update_time) {
    renderCache->lastUsedTime = new_time;
  }
  renderCache->results = std::move(new_results);
  renderCache->resultsRowShape = std::move(new_results_row_shape);
  renderCache->cachedBytes = new_size;
}

TableId NonProjectionRenderQueryCacheMap::_getNextUnusedTableId() {
  auto startId = emptyTableId - 1;  // start at id -2
  auto itr = _queryResultMap.rbegin();
  while (itr != _queryResultMap.rend() && startId == (*itr)->tableId) {
    startId = (*itr)->tableId - 1;
    itr++;
  }
  return startId;
}

void NonProjectionRenderQueryCacheMap::_purgeUnusedCaches() {
  size_t cnt = 0;
  size_t removedBytes = 0;
  auto itr = _queryResultMap.begin();
  for (; itr != _queryResultMap.end();) {
    if (itr->use_count() == 1) {
      // no one else is referencing the cache, so it can be removed
      cnt++;
      removedBytes += (*itr)->cachedBytes;
      itr = _queryResultMap.erase(itr);
    } else {
      itr++;
    }
  }

  LOG_IF(INFO, cnt > 0) << "RootCache - purging " << cnt << " unreferenced render caches equating to " << removedBytes
                        << " bytes of memory being freed.";
  _totalCachedBytes -= removedBytes;
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::addQueryResultToCache(
    const std::string& sqlStr,
    const ResultRows&& results,
    const std::vector<TargetMetaInfo>&& resultRowShape,
    const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery) {
  // TODO(croot): make thread safe?

  _purgeUnusedCaches();

  auto numBytes = results.getQueryMemDesc().getBufferSizeBytes(results.getDeviceType());

  // TODO(croot): if an error is thrown, what happens to results (which should be called with a
  // std::move())
  RUNTIME_EX_ASSERT(numBytes + _totalCachedBytes <= _maxTotalCachedBytes,
                    "Can't add the query results to the render cache for hit-testing. The results are  " +
                        std::to_string(numBytes) + " in size, but there's only " +
                        std::to_string(_maxTotalCachedBytes - _totalCachedBytes) +
                        " unused bytes remaining in the cache.");

  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  auto itr = cacheMapBySql.find(sqlStr);
  RUNTIME_EX_ASSERT(itr == cacheMapBySql.end(),
                    "The sql \"" + sqlStr +
                        "\" already exists in the cache. If there are new results, call "
                        "NonProjectionRenderQueryCacheMap::updateQueryResultsInCache()");

  auto tableId = _getNextUnusedTableId();
  CHECK(_queryResultMap.find(tableId) == _queryResultMap.end());

  auto rtnpair = _queryResultMap.emplace(new NonProjectionRenderQueryCache(
      tableId, sqlStr, numBytes, std::move(results), std::move(resultRowShape), std::move(usedTablesInQuery)));
  CHECK(rtnpair.second);

  _totalCachedBytes += numBytes;

  return *(rtnpair.first);
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::addQueryOnlyToCache(
    const std::string& sqlStr,
    const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery) {
  // TODO(croot): make thread safe?

  _purgeUnusedCaches();

  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  auto itr = cacheMapBySql.find(sqlStr);
  RUNTIME_EX_ASSERT(itr == cacheMapBySql.end(),
                    "The sql \"" + sqlStr +
                        "\" already exists in the cache. If there query is fast and can run standalone on hittests, "
                        "then run NonProjectionRenderQueryCacheMap::removeQueryResultsFromCache()");

  auto tableId = _getNextUnusedTableId();
  CHECK(_queryResultMap.find(tableId) == _queryResultMap.end());

  auto rtnpair = _queryResultMap.emplace(new NonProjectionRenderQueryCache(
      tableId,
      sqlStr,
      0,
      ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, 0),
      std::vector<TargetMetaInfo>(),
      std::move(usedTablesInQuery)));
  CHECK(rtnpair.second);

  return *(rtnpair.first);
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::updateQueryResultsInCache(
    const std::string& sqlStr,
    const ResultRows&& results,
    const std::vector<TargetMetaInfo>&& resultRowShape) {
  // TODO(croot): make thread safe?
  _purgeUnusedCaches();
  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  auto itr = cacheMapBySql.find(sqlStr);

  RUNTIME_EX_ASSERT(itr != cacheMapBySql.end(),
                    "A render query result cache for sql \"" + sqlStr + "\" does not exist.");

  auto numBytes = results.getQueryMemDesc().getBufferSizeBytes(results.getDeviceType());
  auto currBytes = (*itr)->cachedBytes;

  // TODO(croot): if an error is thrown, what happens to results (which should be called with a
  // std::move())
  RUNTIME_EX_ASSERT(_totalCachedBytes - currBytes + numBytes <= _maxTotalCachedBytes,
                    "Can't updated new query results to the render cache for hit-testing. The results are  " +
                        std::to_string(numBytes) + " in size, but there's only " +
                        std::to_string(_maxTotalCachedBytes - _totalCachedBytes + (*itr)->cachedBytes) +
                        " unused bytes remaining in the cache.");

  cacheMapBySql.modify(itr, UpdateCacheResults(std::move(results), std::move(resultRowShape), numBytes));
  _totalCachedBytes += (numBytes - currBytes);

  return *itr;
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::updateQueryResultsInCache(
    const TableId tableId,
    const ResultRows&& results,
    const std::vector<TargetMetaInfo>&& resultRowShape) {
  // TODO(croot): make thread safe?
  _purgeUnusedCaches();
  auto itr = _queryResultMap.find(tableId);

  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id: " + std::to_string(tableId) + " does not exist.");

  auto numBytes = results.getQueryMemDesc().getBufferSizeBytes(results.getDeviceType());
  auto currBytes = (*itr)->cachedBytes;

  // TODO(croot): if an error is thrown, what happens to results (which should be called with a
  // std::move())
  RUNTIME_EX_ASSERT(_totalCachedBytes - currBytes + numBytes <= _maxTotalCachedBytes,
                    "Can't updated new query results to the render cache for hit-testing. The results are  " +
                        std::to_string(numBytes) + " in size, but there's only " +
                        std::to_string(_maxTotalCachedBytes - _totalCachedBytes + (*itr)->cachedBytes) +
                        " unused bytes remaining in the cache.");

  _queryResultMap.modify(itr, UpdateCacheResults(std::move(results), std::move(resultRowShape), numBytes));
  _totalCachedBytes += (numBytes - currBytes);

  return *itr;
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::removeQueryResultsFromCache(const TableId tableId) {
  // TODO(croot): make thread safe?
  _purgeUnusedCaches();
  auto itr = _queryResultMap.find(tableId);

  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id: " + std::to_string(tableId) + " does not exist.");

  auto bytes = (*itr)->cachedBytes;
  _queryResultMap.modify(
      itr,
      UpdateCacheResults(ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, 0),
                         std::vector<TargetMetaInfo>(),
                         0));
  _totalCachedBytes -= bytes;

  return *itr;
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::removeQueryResultsFromCache(const std::string& sqlStr) {
  // TODO(croot): make thread safe?
  _purgeUnusedCaches();
  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  auto itr = cacheMapBySql.find(sqlStr);

  RUNTIME_EX_ASSERT(itr != cacheMapBySql.end(),
                    "A render query result cache for sql \"" + sqlStr + "\" does not exist.");

  auto bytes = (*itr)->cachedBytes;
  cacheMapBySql.modify(
      itr,
      UpdateCacheResults(ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, 0),
                         std::vector<TargetMetaInfo>(),
                         0));

  _totalCachedBytes -= bytes;

  return *itr;
}

bool NonProjectionRenderQueryCacheMap::hasQueryCache(const std::string& sqlStr) {
  _purgeUnusedCaches();
  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  return cacheMapBySql.find(sqlStr) != cacheMapBySql.end();
}

bool NonProjectionRenderQueryCacheMap::hasQueryCache(const TableId tableId) {
  _purgeUnusedCaches();
  return _queryResultMap.find(tableId) != _queryResultMap.end();
}

std::string NonProjectionRenderQueryCacheMap::getQueryForCache(const TableId tableId) const {
  auto itr = _queryResultMap.find(tableId);
  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id " + std::to_string(tableId) + " does not exist.");
  return (*itr)->sqlStr;
}

std::pair<TableId, std::string> NonProjectionRenderQueryCacheMap::getQueryCachePrimaryTableInfo(
    const TableId tableId) const {
  auto itr = _queryResultMap.find(tableId);
  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id " + std::to_string(tableId) + " does not exist.");
  return (*itr)->getPrimaryTableInfo();
}

bool NonProjectionRenderQueryCacheMap::canFitResults(const ResultRows& newResults) {
  _purgeUnusedCaches();
  auto numBytes = newResults.getQueryMemDesc().getBufferSizeBytes(newResults.getDeviceType());
  return _totalCachedBytes + numBytes <= _maxTotalCachedBytes;
}

bool NonProjectionRenderQueryCacheMap::canFitUpdatedResults(const std::string& sqlStr,
                                                            const ResultRows& updatedResults) {
  _purgeUnusedCaches();
  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  auto itr = cacheMapBySql.find(sqlStr);

  RUNTIME_EX_ASSERT(itr != cacheMapBySql.end(),
                    "A render query result cache for sql \"" + sqlStr + "\" does not exist.");

  auto numBytes = updatedResults.getQueryMemDesc().getBufferSizeBytes(updatedResults.getDeviceType());
  return _totalCachedBytes + numBytes - (*itr)->cachedBytes <= _maxTotalCachedBytes;
}

bool NonProjectionRenderQueryCacheMap::canFitUpdatedResults(const TableId tableId, const ResultRows& updatedResults) {
  _purgeUnusedCaches();
  auto itr = _queryResultMap.find(tableId);

  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id: " + std::to_string(tableId) + " does not exist.");

  auto numBytes = updatedResults.getQueryMemDesc().getBufferSizeBytes(updatedResults.getDeviceType());
  return _totalCachedBytes + numBytes - (*itr)->cachedBytes <= _maxTotalCachedBytes;
}

std::pair<const ResultRows*, const std::vector<TargetMetaInfo>*> NonProjectionRenderQueryCacheMap::getQueryCacheResults(
    const TableId tableId) const {
  auto itr = _queryResultMap.find(tableId);
  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id " + std::to_string(tableId) + " does not exist.");
  return std::make_pair(&(*itr)->results, &(*itr)->resultsRowShape);
}

bool RootCache::hasPolyTableGpuCache(const std::string& tableName) {
  return polyCacheMap.find(tableName) != polyCacheMap.end();
}

bool RootCache::hasPolyTableGpuCache(const std::string& tableName, const GpuId gpuId) {
  auto itr = polyCacheMap.find(tableName);
  if (itr == polyCacheMap.end()) {
    return false;
  }

  return itr->second.second.usesGpu(gpuId);
}

bool RootCache::hasPolyTableGpuCache(const std::string& tableName, const std::string& sqlStr) {
  auto itr = polyCacheMap.find(tableName);
  if (itr == polyCacheMap.end()) {
    return false;
  }

  return itr->second.first == buildSqlHash(sqlStr);
}

bool RootCache::hasPolyTableGpuCache(const std::string& tableName, const std::string& sqlStr, const GpuId gpuId) {
  auto itr = polyCacheMap.find(tableName);
  if (itr == polyCacheMap.end()) {
    return false;
  }

  if (itr->second.first != buildSqlHash(sqlStr)) {
    return false;
  }

  return itr->second.second.usesGpu(gpuId);
}

bool RootCache::supportsInt64() const {
  if (!_supportsInt64Ptr) {
    _supportsInt64Ptr.reset(
        new bool(supportedExtensions.find("GL_NV_vertex_attrib_integer_64bit") != supportedExtensions.end()));
  }

  return *_supportsInt64Ptr;
}

}  // namespace QueryRenderer
