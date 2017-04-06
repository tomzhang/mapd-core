#include "RootCache.h"
#include <Rendering/RenderError.h>

namespace QueryRenderer {

template <typename T1, typename T2>
static void inactivateBuffer(std::shared_ptr<QueryBufferPool<T1>>& bufferPool,
                             std::shared_ptr<T2>& buffer,
                             const bool deleteRsrc) {
  std::weak_ptr<T1> wkptr = std::dynamic_pointer_cast<T1>(buffer);
  buffer = nullptr;
  if (deleteRsrc) {
    bufferPool->deleteRsrc(wkptr);
  } else {
    bufferPool->setRsrcInactive(wkptr);
  }
}

const std::chrono::milliseconds SqlPolyQueryCacheMap::maxRsrcIdleTime = std::chrono::milliseconds(300000);

size_t SqlPolyQueryCacheMap::_inactivatePolyDataTableCaches(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                                            BaseQueryPolyDataTable::PerGpuData* cacheBuffers,
                                                            const bool deleteRsrc) {
  size_t numBytes(0);
  if (cacheBuffers->ubo) {
    numBytes += cacheBuffers->ubo->getNumUsedBytes();
    inactivateBuffer(gpuData->uboPool, cacheBuffers->ubo, deleteRsrc);
  }

  if (cacheBuffers->indvbo) {
    numBytes += cacheBuffers->indvbo->getNumUsedBytes();
    inactivateBuffer(gpuData->indvboPool, cacheBuffers->indvbo, deleteRsrc);
  }

  if (cacheBuffers->indibo) {
    numBytes += cacheBuffers->indibo->getNumUsedBytes();
    inactivateBuffer(gpuData->indiboPool, cacheBuffers->indibo, deleteRsrc);
  }

  return numBytes;
}

size_t SqlPolyQueryCacheMap::_inactivatePolyDataTable(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                                      SqlQueryPolyDataTableCacheShPtr& sqlPolyDataTablePtr,
                                                      const bool deleteRsrc) {
  auto cacheBuffers = &(sqlPolyDataTablePtr->getQueryBuffers(gpuData->getGpuId()));
  auto rtn = _inactivatePolyDataTableCaches(gpuData, cacheBuffers, deleteRsrc);
  sqlPolyDataTablePtr->clear();
  return rtn;
}

size_t SqlPolyQueryCacheMap::_inactivatePolyVertCache(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                                      std::shared_ptr<PolyVertCache>& polyVertCache,
                                                      const bool deleteRsrc) {
  size_t numBytes = 0;
  for (auto& item : polyVertCache->sqlCacheMap) {
    auto ptr = item.sqlPolyDataTablePtr;
    numBytes += _inactivatePolyDataTable(gpuData, ptr, deleteRsrc);
  }

  if (polyVertCache->vbo && polyVertCache->vbo.use_count() == 2) {
    numBytes += polyVertCache->vbo->getNumUsedBytes();
    inactivateBuffer(gpuData->vboPool, polyVertCache->vbo, deleteRsrc);
  }

  if (polyVertCache->ibo && polyVertCache->ibo.use_count() == 2) {
    numBytes += polyVertCache->ibo->getNumUsedBytes();
    inactivateBuffer(gpuData->iboPool, polyVertCache->ibo, deleteRsrc);
  }

  return numBytes;
}

bool SqlPolyQueryCacheMap::_cleanOldCachesToMakeRoom(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                                     PolyCacheMap* polyTableMap,
                                                     const size_t maxNumBytes,
                                                     const size_t requiredNumBytes,
                                                     const std::chrono::milliseconds& cutoffTime,
                                                     const std::string* polyTableNameToKeep,
                                                     const bool force) {
  if (!force && gpuData->totalUsedBytes + requiredNumBytes <= maxNumBytes) {
    return true;
  }
  auto diff = gpuData->totalUsedBytes + requiredNumBytes - maxNumBytes;

  size_t numBytes(0);
  std::vector<PolyCacheMap::iterator> polyCaches;
  std::unordered_map<SqlCache_by_lastTime*, std::pair<PolyVertCache*, std::vector<SqlCache_by_lastTime::iterator>>>
      sqlCaches;
  for (auto polycacheitr = gpuData->polyCacheMap.begin(); polycacheitr != gpuData->polyCacheMap.end(); ++polycacheitr) {
    if ((!polyTableNameToKeep || *polyTableNameToKeep != (*polycacheitr)->polyTableName) &&
        (*polycacheitr)->getLastUpdateTime() < cutoffTime) {
      numBytes += (*polycacheitr)->totalUsedBytes;
      polyCaches.push_back(polycacheitr);
    } else {
      auto& sqlLookup = (*polycacheitr)->sqlCacheMap.get<CacheLastUsedTimeTag>();
      for (auto sqlcacheitr = sqlLookup.begin(); sqlcacheitr != sqlLookup.end(); ++sqlcacheitr) {
        if (sqlcacheitr->lastUsedTime < cutoffTime) {
          numBytes += sqlcacheitr->totalUsedBytes;
          auto sqlcacheptritr = sqlCaches.find(&sqlLookup);
          if (sqlcacheptritr == sqlCaches.end()) {
            auto insertpair = sqlCaches.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(&sqlLookup),
                std::forward_as_tuple(
                    std::piecewise_construct, std::forward_as_tuple(polycacheitr->get()), std::forward_as_tuple()));
            CHECK(insertpair.second);
            sqlcacheptritr = insertpair.first;
          }
          sqlcacheptritr->second.second.push_back(sqlcacheitr);
          if (!force && numBytes >= diff) {
            break;
          }
        } else {
          break;
        }
      }
    }
    if (!force && numBytes >= diff) {
      break;
    }
  }

  if (force || numBytes >= diff) {
    if (polyCaches.size() || sqlCaches.size()) {
      gpuData->makeActiveOnCurrentThread();
      numBytes = 0;
      for (auto& itr : polyCaches) {
        polyTableMap->erase((*itr)->polyTableName);
        auto ptr = *itr;
        numBytes += _inactivatePolyVertCache(gpuData, ptr, true);

        LOG(INFO) << "Purging all poly caches on gpu " << gpuData->getGpuId() << " for table: " << (*itr)->polyTableName
                  << " clearing up " << numBytes << " bytes. All caches have been idle for too long.";
      }

      for (auto& itr : sqlCaches) {
        for (auto& sqlitr : itr.second.second) {
          auto ptr = sqlitr->sqlPolyDataTablePtr;
          auto sqlNumBytes = _inactivatePolyDataTable(gpuData, ptr, true);
          LOG(INFO) << "Purging poly cache: " << ptr->getSqlQueryStrRef() << " on gpu " << gpuData->getGpuId()
                    << ". Clearing up " << sqlNumBytes << " bytes. Cache has been idle too long.";
          CHECK(sqlNumBytes <= itr.second.first->totalUsedBytes);
          itr.first->erase(sqlitr);
          itr.second.first->totalUsedBytes -= sqlNumBytes;
          numBytes += sqlNumBytes;
        }
      }
      CHECK(numBytes <= gpuData->totalUsedBytes);
      gpuData->totalUsedBytes -= numBytes;
      gpuData->makeInactive();
    }
    return true;
  }
  return false;
}

void SqlPolyQueryCacheMap::_clearOverflowBuffers(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                                 PolyCacheMap& polyTableMap) {
  for (auto& item : gpuData->overflowPolyMap) {
    auto ptr = item;
    gpuData->makeActiveOnCurrentThread();
    _inactivatePolyVertCache(gpuData, ptr, false);
    PolyCache_by_ptr& polyLookup = polyTableMap.get<CachePolyPtrTag>();
    if (polyLookup.erase(item)) {
      auto itr = gpuData->polyCacheMap.find(item->polyTableName);
      if (itr != gpuData->polyCacheMap.end()) {
        polyTableMap.insert(*itr);
      }
    }
    gpuData->makeInactive();
  }
  gpuData->overflowPolyMap.clear();
}

SqlPolyQueryCacheMap::SqlPolyQueryCacheMap(RootCache* prntGpuCache, const size_t maxCacheBytes)
    : prntGpuCache(prntGpuCache), _maxCacheBytes(maxCacheBytes) {
  for (auto& item : (*prntGpuCache->perGpuData)) {
    _polyCaches.emplace(std::piecewise_construct, std::forward_as_tuple(item->gpuId), std::forward_as_tuple(item));
  }
}

GpuId SqlPolyQueryCacheMap::getGpuIdForPolyTableCache(const std::string& polyTableName) {
  // clear out any old unused caches
  std::chrono::milliseconds cutoffTime = getCurrentTimeMS() - maxRsrcIdleTime;
  for (auto& item : _polyCaches) {
    _cleanOldCachesToMakeRoom(&(item.second), &_polyTableMap, _maxCacheBytes, 0, cutoffTime, &polyTableName, true);
  }

  PerGpuData* gpuData = nullptr;
  auto itr = _polyTableMap.find(polyTableName);
  if (itr == _polyTableMap.end()) {
    // find gpu id to use for this new cache
    // TODO(croot): query for avail mem on each gpu?
    decltype(_polyCaches)::iterator perGpuItr;
    if (_polyCaches.size() == 1) {
      perGpuItr = _polyCaches.begin();
    } else {
      perGpuItr = std::min_element(_polyCaches.begin(), _polyCaches.end(), [](auto& item1, auto& item2) {
        return item1.second.totalUsedBytes < item2.second.totalUsedBytes;
      });
    }
    CHECK(perGpuItr != _polyCaches.end());
    gpuData = &(perGpuItr->second);
  } else {
    gpuData = (*itr)->perGpuDataPtr;
  }
  CHECK(gpuData);
  return gpuData->getGpuId();
}

void SqlPolyQueryCacheMap::updateCache(const RootCacheShPtr& qrmGpuCache,
                                       const GpuId gpuId,
                                       const std::string& polyTableName,
                                       const std::string& queryStr,
                                       const PolyTableByteData& initTableData,
                                       const QueryDataLayoutShPtr& vertLayoutPtr,
                                       const PolyRowDataShPtr& rowDataPtr,
                                       const QueryDataLayoutShPtr& uniformLayoutPtr) {
  CHECK(qrmGpuCache.get() == prntGpuCache);

  auto gpuitr = _polyCaches.find(gpuId);
  CHECK(gpuitr != _polyCaches.end());
  auto& gpuData = gpuitr->second;

  auto itr = _polyTableMap.find(polyTableName);
  auto uniqueNumBytes = std::max(initTableData.numDataBytes, 0) + std::max(initTableData.numLineLoopBytes, 0) +
                        std::max(initTableData.numPolyBytes, 0);
  auto totalNumBytes =
      uniqueNumBytes + std::max(initTableData.numVertBytes, 0) + std::max(initTableData.numIndexBytes, 0);

  PolyVertCache* polyCache = nullptr;
  SqlQueryPolyDataTableCache* sqlPolyCache = nullptr;
  PolySqlCacheMap::iterator sqlitr;

  bool useCacheBuffers = true;

  if (itr == _polyTableMap.end()) {
    RUNTIME_EX_ASSERT(vertLayoutPtr,
                      "Cannot create a poly query cache for table " + polyTableName + " without a vertex layout ptr");

    if (gpuData.totalUsedBytes + totalNumBytes > _maxCacheBytes) {
      useCacheBuffers = false;
    }

    PolyCacheMap* polyCacheMapToUse;
    if (useCacheBuffers) {
      polyCacheMapToUse = &(gpuData.polyCacheMap);
    } else {
      polyCacheMapToUse = &(gpuData.overflowPolyMap);
    }

    auto polyemplaceitr = polyCacheMapToUse->emplace(new PolyVertCache(&gpuData, polyTableName, !useCacheBuffers));
    CHECK(polyemplaceitr.second);
    auto insertrtn = _polyTableMap.insert(*(polyemplaceitr.first));
    CHECK(insertrtn.second);
    polyCache = (*(polyemplaceitr.first)).get();
    auto sqlemplaceitr = polyCache->sqlCacheMap.emplace(
        SqlQueryPolyDataTableCacheShPtr(new SqlQueryPolyDataTableCache(qrmGpuCache, polyTableName, queryStr)));
    CHECK(sqlemplaceitr.second);
    sqlitr = sqlemplaceitr.first;

    LOG_IF(INFO, useCacheBuffers) << "Caching " << totalNumBytes << " bytes (" << (totalNumBytes - uniqueNumBytes)
                                  << " for vbo/ibo) for poly query: " << queryStr << " on gpu " << gpuId;
  } else {
    CHECK((*itr)->perGpuDataPtr == &gpuData);

    PolyCache_by_ptr& cachePolyLookup = gpuData.polyCacheMap.get<CachePolyPtrTag>();
    PolyCache_by_ptr& overflowPolyLookup = gpuData.overflowPolyMap.get<CachePolyPtrTag>();

    auto cacheitr = cachePolyLookup.find(*itr);
    auto overflowitr = overflowPolyLookup.find(*itr);
    decltype(cacheitr)* itrtouse = &cacheitr;
    useCacheBuffers = (cacheitr != cachePolyLookup.end());
    if (!useCacheBuffers) {
      CHECK(overflowitr != overflowPolyLookup.end() && (*itr)->isOverflow);
      itrtouse = &overflowitr;
    }

    polyCache = (*itrtouse)->get();
    auto sqlmap = &polyCache->sqlCacheMap;

    SqlCache_by_sql* sqlLookup = &(sqlmap->get<CacheSqlTag>());
    auto sqlcacheitr = sqlLookup->find(queryStr);
    if (sqlcacheitr == sqlLookup->end()) {
      if (useCacheBuffers && gpuData.totalUsedBytes + uniqueNumBytes > _maxCacheBytes) {
        useCacheBuffers = false;

        if (overflowitr == overflowPolyLookup.end()) {
          auto polyemplaceitr = gpuData.overflowPolyMap.emplace(new PolyVertCache(&gpuData, polyTableName, true));
          CHECK(polyemplaceitr.second);
          (*polyemplaceitr.first)->vbo = polyCache->vbo;
          (*polyemplaceitr.first)->ibo = polyCache->ibo;
          polyCache = polyemplaceitr.first->get();
        } else {
          polyCache = overflowitr->get();
        }
        sqlmap = &polyCache->sqlCacheMap;
        sqlLookup = &(sqlmap->get<CacheSqlTag>());
        CHECK(sqlLookup->find(queryStr) == sqlLookup->end());
      }

      auto sqlemplaceitr = sqlmap->emplace(
          SqlQueryPolyDataTableCacheShPtr(new SqlQueryPolyDataTableCache(qrmGpuCache, polyTableName, queryStr)));
      CHECK(sqlemplaceitr.second);
      sqlitr = sqlemplaceitr.first;

      LOG_IF(INFO, useCacheBuffers) << "Caching " << uniqueNumBytes << " bytes for poly query: " << queryStr
                                    << " on gpu " << gpuId;
    } else {
      sqlLookup->modify(sqlcacheitr, ChangeLastUsedTime());
      sqlitr = sqlmap->project<CachePtrTag>(sqlcacheitr);
    }
  }

  sqlPolyCache = sqlitr->sqlPolyDataTablePtr.get();
  CHECK(polyCache && sqlPolyCache);
  auto cachedBuffers = &(sqlPolyCache->setUsedGpu(gpuData.getRootPerGpuData()));

  CHECK(cachedBuffers && (!cachedBuffers->vbo || cachedBuffers->vbo == polyCache->vbo) &&
        (!cachedBuffers->ibo || cachedBuffers->ibo == polyCache->ibo));

  auto currUniqueNumBytes = sqlitr->totalUsedBytes;

  auto vboSize = (polyCache->vbo ? polyCache->vbo->getNumUsedBytes() : 0);
  auto iboSize = (polyCache->ibo ? polyCache->ibo->getNumUsedBytes() : 0);
  auto currNumBytes = vboSize + iboSize + currUniqueNumBytes;

  if (initTableData.numVertBytes < 0) {
    totalNumBytes += vboSize;
  }
  if (initTableData.numIndexBytes < 0) {
    totalNumBytes += iboSize;
  }
  if (initTableData.numDataBytes < 0) {
    auto bytes = (cachedBuffers->ubo ? cachedBuffers->ubo->getNumUsedBytes() : 0);
    totalNumBytes += bytes;
    uniqueNumBytes += bytes;
  }
  if (initTableData.numLineLoopBytes < 0) {
    auto bytes = (cachedBuffers->indvbo ? cachedBuffers->indvbo->getNumUsedBytes() : 0);
    totalNumBytes += bytes;
    uniqueNumBytes += bytes;
  }
  if (initTableData.numPolyBytes < 0) {
    auto bytes = (cachedBuffers->indibo ? cachedBuffers->indibo->getNumUsedBytes() : 0);
    totalNumBytes += bytes;
    uniqueNumBytes += bytes;
  }

  int64_t uniqueByteDiff = static_cast<int64_t>(uniqueNumBytes) - static_cast<int64_t>(currUniqueNumBytes);
  int64_t totalByteDiff = static_cast<int64_t>(totalNumBytes) - static_cast<int64_t>(currNumBytes);

  if (useCacheBuffers && gpuData.totalUsedBytes + totalByteDiff > _maxCacheBytes) {
    // If we reach here it means that the poly table and query are found in the cache,
    // but its buffers are being resized for some reason (i.e. the verts changed or something)
    // causing a resize event, and the buffer resize exceeds the cache limit.
    THROW_RUNTIME_EX("Resizing poly buffers not supported yet");
    useCacheBuffers = false;
  }

  gpuData.makeActiveOnCurrentThread();

  if (initTableData.numVertBytes > 0) {
    if (!polyCache->vbo) {
      polyCache->vbo = gpuData.vboPool->getInactiveRsrc(initTableData.numVertBytes).lock();
    } else {
      polyCache->vbo->resize(initTableData.numVertBytes);
    }
  } else if (initTableData.numVertBytes == 0) {
    // not deleting, because all references would need to be deleted
    // This ensures all references get a zero-length buffer
    if (polyCache->vbo) {
      polyCache->vbo->resize(0);
    }
  }

  if (vertLayoutPtr && polyCache->vbo) {
    polyCache->vbo->setBufferLayout(
        vertLayoutPtr->convertToBufferLayout(qrmGpuCache->supportedExtensions), initTableData.numVertBytes, 0);
  }
  cachedBuffers->vbo = polyCache->vbo;

  if (initTableData.numIndexBytes > 0) {
    if (!polyCache->ibo) {
      polyCache->ibo = gpuData.iboPool->getInactiveRsrc(initTableData.numIndexBytes).lock();
    } else {
      polyCache->ibo->resize(initTableData.numIndexBytes);
    }
  } else if (initTableData.numIndexBytes == 0) {
    if (polyCache->ibo) {
      polyCache->ibo->resize(0);
    }
  }
  cachedBuffers->ibo = polyCache->ibo;

  if (initTableData.numDataBytes > 0) {
    if (!cachedBuffers->ubo) {
      cachedBuffers->ubo = gpuData.uboPool->getInactiveRsrc(initTableData.numDataBytes).lock();
    } else {
      cachedBuffers->ubo->resize(initTableData.numDataBytes);
    }
  } else if (initTableData.numDataBytes == 0) {
    inactivateBuffer(gpuData.uboPool, cachedBuffers->ubo, true);
  }

  if (uniformLayoutPtr && cachedBuffers->ubo) {
    cachedBuffers->ubo->setBufferLayout(
        uniformLayoutPtr->convertToUniformBufferLayout(prntGpuCache->supportedExtensions),
        initTableData.numDataBytes,
        0);
  }

  if (initTableData.numLineLoopBytes > 0) {
    if (!cachedBuffers->indvbo) {
      cachedBuffers->indvbo = gpuData.indvboPool->getInactiveRsrc(initTableData.numLineLoopBytes).lock();
    } else {
      cachedBuffers->indvbo->resize(initTableData.numLineLoopBytes);
    }
  } else if (initTableData.numLineLoopBytes == 0) {
    inactivateBuffer(gpuData.indvboPool, cachedBuffers->indvbo, true);
  }

  if (initTableData.numPolyBytes > 0) {
    if (!cachedBuffers->indibo) {
      cachedBuffers->indibo = gpuData.indiboPool->getInactiveRsrc(initTableData.numPolyBytes).lock();
    } else {
      cachedBuffers->indibo->resize(initTableData.numPolyBytes);
    }
  } else if (initTableData.numPolyBytes == 0) {
    inactivateBuffer(gpuData.indiboPool, cachedBuffers->indibo, true);
  }

  // NOTE: managing the counts here can be dangerous if the underlying
  // buffers are changed externally somewhere. Currently there's no where
  // in the code that does this other than here, but if it does happen
  // at some point, then this can get out of sync quick.
  polyCache->sqlCacheMap.modify(sqlitr, ChangeTotalUsedBytes(uniqueByteDiff));
  if (useCacheBuffers) {
    gpuData.totalUsedBytes += totalByteDiff;
    polyCache->totalUsedBytes += totalByteDiff;
  }
  polyCache->rowDataPtr = rowDataPtr;
  cachedBuffers->rowDataPtr = rowDataPtr;

  gpuData.makeInactive();
}

void SqlPolyQueryCacheMap::clearOverflowBuffers() {
  for (auto& item : _polyCaches) {
    _clearOverflowBuffers(&(item.second), _polyTableMap);
  }
}

void SqlPolyQueryCacheMap::deleteCache(const std::string& polyTableName, const std::string& queryStr) {
  auto polyitr = _polyTableMap.find(polyTableName);
  if (polyitr == _polyTableMap.end()) {
    return;
  }

  auto gpuPtr = (*polyitr)->perGpuDataPtr;
  CHECK(gpuPtr);
  auto itr = gpuPtr->polyCacheMap.find(polyTableName);
  CHECK(itr != gpuPtr->polyCacheMap.end());
  gpuPtr->makeActiveOnCurrentThread();

  size_t numBytes(0);
  if (queryStr.length()) {
    SqlCache_by_sql& sqlLookup = (*itr)->sqlCacheMap.get<CacheSqlTag>();
    auto sqlitr = sqlLookup.find(queryStr);
    if (sqlitr == sqlLookup.end()) {
      return;
    }

    auto ptr = sqlitr->sqlPolyDataTablePtr;
    numBytes = _inactivatePolyDataTable(gpuPtr, ptr, true);
    LOG(INFO) << "Purging poly caches for query: " << queryStr << " on gpu " << gpuPtr->getGpuId() << ". Clearing up "
              << numBytes << " bytes.";
    CHECK(numBytes == sqlitr->totalUsedBytes && numBytes <= (*itr)->totalUsedBytes);
    sqlLookup.erase(sqlitr);
    (*itr)->totalUsedBytes -= numBytes;
  } else {
    auto ptr = *itr;
    numBytes = _inactivatePolyVertCache(gpuPtr, ptr, true);
    LOG(INFO) << "Purging poly caches for all queryies for table: " << polyTableName << " on gpu " << gpuPtr->getGpuId()
              << ". Clearing up " << numBytes << " bytes.";
    _polyTableMap.erase(itr);
    gpuPtr->polyCacheMap.erase(polyTableName);
  }

  CHECK(numBytes <= gpuPtr->totalUsedBytes);
  gpuPtr->totalUsedBytes -= numBytes;

  gpuPtr->makeInactive();
}

std::tuple<bool, SqlPolyQueryCacheMap::PolyCacheMap::iterator, SqlPolyQueryCacheMap::SqlCache_by_sql::iterator>
SqlPolyQueryCacheMap::_getSqlItr(const SqlPolyQueryCacheMap::PolyCacheMap& polyMap,
                                 const std::string& polyTableName,
                                 const std::string& queryStr) {
  auto polyitr = polyMap.find(polyTableName);
  if (polyitr == polyMap.end()) {
    return std::make_tuple(false, polyitr, SqlCache_by_sql::iterator());
  }

  auto& sqlLookup = (*polyitr)->sqlCacheMap.get<CacheSqlTag>();
  auto sqlitr = sqlLookup.find(queryStr);
  return std::make_tuple(sqlitr != sqlLookup.end(), polyitr, sqlitr);
}

std::tuple<bool, SqlPolyQueryCacheMap::PolyCacheMap::iterator, SqlPolyQueryCacheMap::SqlCache_by_sql::iterator>
SqlPolyQueryCacheMap::_getSqlItr(SqlPolyQueryCacheMap::PerGpuData* gpuData,
                                 const std::string& polyTableName,
                                 const std::string& queryStr) {
  auto rtn = _getSqlItr(gpuData->polyCacheMap, polyTableName, queryStr);
  if (std::get<0>(rtn)) {
    return rtn;
  }
  return _getSqlItr(gpuData->overflowPolyMap, polyTableName, queryStr);
}

PolyTableByteData SqlPolyQueryCacheMap::getPolyBufferByteData(GpuId gpuId,
                                                              const std::string& polyTableName,
                                                              const std::string& queryStr) {
  auto itr = _polyCaches.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _polyCaches.end(), "Cannot find poly cached data for gpuId " + std::to_string(gpuId));

  bool found;
  PolyCacheMap::iterator polyitr;
  SqlCache_by_sql::iterator sqlitr;
  std::tie(found, polyitr, sqlitr) = _getSqlItr(&(itr->second), polyTableName, queryStr);
  RUNTIME_EX_ASSERT(found, "Cannot find poly cached data for table " + polyTableName + " with sql " + queryStr);

  return sqlitr->sqlPolyDataTablePtr->getPolyBufferByteData(gpuId);
}

PolyTableDataInfo SqlPolyQueryCacheMap::getPolyBufferData(GpuId gpuId,
                                                          const std::string& polyTableName,
                                                          const std::string& queryStr) {
  auto itr = _polyCaches.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _polyCaches.end(), "Cannot find poly cached data for gpuId " + std::to_string(gpuId));

  bool found;
  PolyCacheMap::iterator polyitr;
  SqlCache_by_sql::iterator sqlitr;
  std::tie(found, polyitr, sqlitr) = _getSqlItr(&(itr->second), polyTableName, queryStr);
  RUNTIME_EX_ASSERT(found, "Cannot find poly cached data for table " + polyTableName + " with sql " + queryStr);

  return sqlitr->sqlPolyDataTablePtr->getPolyBufferData(gpuId);
}

PolyCudaHandles SqlPolyQueryCacheMap::getCudaHandlesPreQuery(const RootCacheShPtr& qrmGpuCache,
                                                             GpuId gpuId,
                                                             const std::string& polyTableName,
                                                             const std::string& queryStr) {
  auto itr = _polyCaches.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _polyCaches.end(), "Cannot find poly cached data for gpuId " + std::to_string(gpuId));

  bool found;
  PolyCacheMap::iterator polyitr;
  SqlCache_by_sql::iterator sqlitr;
  std::tie(found, polyitr, sqlitr) = _getSqlItr(&(itr->second), polyTableName, queryStr);
  RUNTIME_EX_ASSERT(found, "Cannot find poly cached data for table " + polyTableName + " with sql " + queryStr);

// TODO(croot): update the last update time?
#ifdef HAVE_CUDA
  return sqlitr->sqlPolyDataTablePtr->getCudaHandlesPreQuery(gpuId);
#else
  return PolyCudaHandles();
#endif
}

void SqlPolyQueryCacheMap::updateCachePostQuery(GpuId gpuId,
                                                const std::string& polyTableName,
                                                const std::string& queryStr,
                                                const QueryDataLayoutShPtr& uniformLayoutPtr,
                                                const QueryDataLayoutShPtr& vertLayoutPtr,
                                                const PolyRowDataShPtr& rowDataPtr) {
  auto itr = _polyCaches.find(gpuId);
  RUNTIME_EX_ASSERT(itr != _polyCaches.end(), "Cannot find poly cached data for gpuId " + std::to_string(gpuId));

  bool found;
  PolyCacheMap::iterator polyitr;
  SqlCache_by_sql::iterator sqlitr;
  std::tie(found, polyitr, sqlitr) = _getSqlItr(&(itr->second), polyTableName, queryStr);
  RUNTIME_EX_ASSERT(found, "Cannot find poly cached data for table " + polyTableName + " with sql " + queryStr);

  if (rowDataPtr) {
    (*polyitr)->rowDataPtr = rowDataPtr;
  }

  sqlitr->sqlPolyDataTablePtr->updatePostQuery(gpuId, vertLayoutPtr, uniformLayoutPtr, rowDataPtr);

  auto& sqlLookup = (*polyitr)->sqlCacheMap.get<CacheSqlTag>();
  sqlLookup.modify(sqlitr, ChangeLastUsedTime());
}

SqlQueryPolyDataTableCacheShPtr SqlPolyQueryCacheMap::getPolyTableCache(const std::string& polyTableName,
                                                                        const std::string& queryStr) {
  auto itr = _polyTableMap.find(polyTableName);
  RUNTIME_EX_ASSERT(itr != _polyTableMap.end() && itr->use_count() == 2 && (*itr)->perGpuDataPtr,
                    "Cannot find poly cached data for table " + polyTableName);

  bool found;
  PolyCacheMap::iterator polyitr;
  SqlCache_by_sql::iterator sqlitr;
  std::tie(found, polyitr, sqlitr) = _getSqlItr((*itr)->perGpuDataPtr, polyTableName, queryStr);
  RUNTIME_EX_ASSERT(found, "Cannot find poly cached data for table " + polyTableName + " with sql " + queryStr);

  return sqlitr->sqlPolyDataTablePtr;
}

void SqlPolyQueryCacheMap::notifyCacheRelease(const SqlQueryPolyDataTableCacheShPtr& releasedCachePtr,
                                              const std::string* swapQueryStr) {
  if (!releasedCachePtr) {
    return;
  }

  auto& polyTableName = releasedCachePtr->getPolyTableNameRef();
  auto polyitr = _polyTableMap.find(polyTableName);
  if (polyitr == _polyTableMap.end()) {
    return;
  }

  auto gpuPtr = (*polyitr)->perGpuDataPtr;
  CHECK(gpuPtr);
  auto itr = gpuPtr->polyCacheMap.find(polyTableName);
  if (itr == gpuPtr->polyCacheMap.end()) {
    return;
  }

  auto& queryStr = releasedCachePtr->getSqlQueryStrRef();
  SqlCache_by_sql& sqlLookup = (*itr)->sqlCacheMap.get<CacheSqlTag>();
  auto sqlitr = sqlLookup.find(queryStr);
  if (sqlitr == sqlLookup.end()) {
    return;
  }

  if (sqlitr->sqlPolyDataTablePtr.use_count() == 2) {
    // this means that releasedCachePtr is the last reference to the cache, so the cache can
    // be removed or swapped for another pending query.
    // If this is the last existing cache for the poly table, then we can
    // remove the full cache for the poly table.
    size_t numBytes;

    gpuPtr->makeActiveOnCurrentThread();

    auto ptr = sqlitr->sqlPolyDataTablePtr;
    if (!swapQueryStr) {
      numBytes = _inactivatePolyDataTable(gpuPtr, ptr, true);
    } else {
      numBytes = _inactivatePolyDataTable(gpuPtr, ptr, false);
    }
    CHECK(numBytes == sqlitr->totalUsedBytes);
    sqlLookup.erase(sqlitr);

    if (sqlLookup.size() == 0 && !swapQueryStr) {
      auto ptr = *itr;
      numBytes += _inactivatePolyVertCache(gpuPtr, ptr, true);
      LOG(INFO) << "Purging poly caches for table: " << polyTableName << " on gpu " << gpuPtr->getGpuId()
                << " clearing up " << numBytes << " bytes. There are no more active sessions rendering this table.";
      _polyTableMap.erase(polyitr);
      gpuPtr->polyCacheMap.erase(itr);
      CHECK(gpuPtr->overflowPolyMap.find(polyTableName) == gpuPtr->overflowPolyMap.end());
    } else {
      LOG(INFO) << "Purging poly caches for query: " << queryStr << " on gpu " << gpuPtr->getGpuId() << " clearing up "
                << numBytes << " bytes. There are no more active sessions rendering this query.";
      CHECK(numBytes <= (*itr)->totalUsedBytes);
      (*itr)->totalUsedBytes -= numBytes;
    }

    CHECK(numBytes <= gpuPtr->totalUsedBytes);
    gpuPtr->totalUsedBytes -= numBytes;

    gpuPtr->makeInactive();
  }
}

bool SqlPolyQueryCacheMap::hasCache(const std::string& tableName) const {
  auto itr = _polyTableMap.find(tableName);
  return itr != _polyTableMap.end() && itr->use_count() == 2 && (*itr)->perGpuDataPtr &&
         ((*itr)->perGpuDataPtr->polyCacheMap.find(tableName) != (*itr)->perGpuDataPtr->polyCacheMap.end() ||
          (*itr)->perGpuDataPtr->overflowPolyMap.find(tableName) != (*itr)->perGpuDataPtr->overflowPolyMap.end());
}

bool SqlPolyQueryCacheMap::hasCache(const std::string& tableName, const GpuId gpuId) const {
  auto itr = _polyCaches.find(gpuId);
  if (itr == _polyCaches.end()) {
    return false;
  }

  return itr->second.polyCacheMap.find(tableName) != itr->second.polyCacheMap.end() ||
         itr->second.overflowPolyMap.find(tableName) != itr->second.overflowPolyMap.end();
}

bool SqlPolyQueryCacheMap::hasCache(const std::string& tableName, const std::string& queryStr) const {
  auto itr = _polyTableMap.find(tableName);
  if (itr == _polyTableMap.end() || itr->use_count() != 2 || !(*itr)->perGpuDataPtr) {
    return false;
  }

  bool found;
  PolyCacheMap::iterator polyitr;
  SqlCache_by_sql::iterator sqlitr;
  std::tie(found, polyitr, sqlitr) = _getSqlItr((*itr)->perGpuDataPtr->polyCacheMap, tableName, queryStr);
  if (!found) {
    std::tie(found, polyitr, sqlitr) = _getSqlItr((*itr)->perGpuDataPtr->overflowPolyMap, tableName, queryStr);
  }
  return found;
}

bool SqlPolyQueryCacheMap::hasCache(const std::string& tableName,
                                    const std::string& queryStr,
                                    const GpuId gpuId) const {
  auto itr = _polyCaches.find(gpuId);
  if (itr == _polyCaches.end()) {
    return false;
  }

  bool found;
  PolyCacheMap::iterator polyitr;
  SqlCache_by_sql::iterator sqlitr;
  std::tie(found, polyitr, sqlitr) = _getSqlItr(itr->second.polyCacheMap, tableName, queryStr);
  if (!found) {
    std::tie(found, polyitr, sqlitr) = _getSqlItr(itr->second.overflowPolyMap, tableName, queryStr);
  }
  return found;
}

const uint16_t NonProjectionRenderQueryCacheMap::maxQueryTime = 250;  // in ms

const std::chrono::milliseconds NonProjectionRenderQueryCacheMap::maxCacheIdleTime =
    std::chrono::milliseconds(300000);  // 5 minutes, in ms
const TableId NonProjectionRenderQueryCacheMap::emptyTableId = -1;

NonProjectionRenderQueryCacheMap::UpdateCacheResults::UpdateCacheResults(
    const std::shared_ptr<ResultRows>& results,
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
    const std::shared_ptr<ResultRows>& results,
    const std::vector<TargetMetaInfo>&& resultRowShape,
    const std::vector<std::pair<TableId, std::string>>&& usedTablesInQuery) {
  // TODO(croot): make thread safe?

  _purgeUnusedCaches();

  auto numBytes = results->getQueryMemDesc().getBufferSizeBytes(results->getDeviceType());

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
      tableId, sqlStr, numBytes, results, std::move(resultRowShape), std::move(usedTablesInQuery)));
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
      std::shared_ptr<ResultRows>(
          new ResultRows({}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, 0)),
      std::vector<TargetMetaInfo>(),
      std::move(usedTablesInQuery)));
  CHECK(rtnpair.second);

  return *(rtnpair.first);
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::updateQueryResultsInCache(
    const std::string& sqlStr,
    const std::shared_ptr<ResultRows>& results,
    const std::vector<TargetMetaInfo>&& resultRowShape) {
  // TODO(croot): make thread safe?
  _purgeUnusedCaches();
  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  auto itr = cacheMapBySql.find(sqlStr);

  RUNTIME_EX_ASSERT(itr != cacheMapBySql.end(),
                    "A render query result cache for sql \"" + sqlStr + "\" does not exist.");

  auto numBytes = results->getQueryMemDesc().getBufferSizeBytes(results->getDeviceType());
  auto currBytes = (*itr)->cachedBytes;

  // TODO(croot): if an error is thrown, what happens to results (which should be called with a
  // std::move())
  RUNTIME_EX_ASSERT(_totalCachedBytes - currBytes + numBytes <= _maxTotalCachedBytes,
                    "Can't updated new query results to the render cache for hit-testing. The results are  " +
                        std::to_string(numBytes) + " in size, but there's only " +
                        std::to_string(_maxTotalCachedBytes - _totalCachedBytes + (*itr)->cachedBytes) +
                        " unused bytes remaining in the cache.");

  cacheMapBySql.modify(itr, UpdateCacheResults(results, std::move(resultRowShape), numBytes));
  _totalCachedBytes += (numBytes - currBytes);

  return *itr;
}

NPRQueryCacheShPtr NonProjectionRenderQueryCacheMap::updateQueryResultsInCache(
    const TableId tableId,
    const std::shared_ptr<ResultRows>& results,
    const std::vector<TargetMetaInfo>&& resultRowShape) {
  // TODO(croot): make thread safe?
  _purgeUnusedCaches();
  auto itr = _queryResultMap.find(tableId);

  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id: " + std::to_string(tableId) + " does not exist.");

  auto numBytes = results->getQueryMemDesc().getBufferSizeBytes(results->getDeviceType());
  auto currBytes = (*itr)->cachedBytes;

  // TODO(croot): if an error is thrown, what happens to results (which should be called with a
  // std::move())
  RUNTIME_EX_ASSERT(_totalCachedBytes - currBytes + numBytes <= _maxTotalCachedBytes,
                    "Can't updated new query results to the render cache for hit-testing. The results are  " +
                        std::to_string(numBytes) + " in size, but there's only " +
                        std::to_string(_maxTotalCachedBytes - _totalCachedBytes + (*itr)->cachedBytes) +
                        " unused bytes remaining in the cache.");

  _queryResultMap.modify(itr, UpdateCacheResults(results, std::move(resultRowShape), numBytes));
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
      UpdateCacheResults(std::shared_ptr<ResultRows>(new ResultRows(
                             {}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, 0)),
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
      UpdateCacheResults(std::shared_ptr<ResultRows>(new ResultRows(
                             {}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU, nullptr, 0, 0, 0, 0)),
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

bool NonProjectionRenderQueryCacheMap::canFitResults(const std::shared_ptr<ResultRows>& newResults) {
  _purgeUnusedCaches();
  auto numBytes = newResults->getQueryMemDesc().getBufferSizeBytes(newResults->getDeviceType());
  return _totalCachedBytes + numBytes <= _maxTotalCachedBytes;
}

bool NonProjectionRenderQueryCacheMap::canFitUpdatedResults(const std::string& sqlStr,
                                                            const std::shared_ptr<ResultRows>& updatedResults) {
  _purgeUnusedCaches();
  auto& cacheMapBySql = _queryResultMap.get<SqlStrTag>();
  auto itr = cacheMapBySql.find(sqlStr);

  RUNTIME_EX_ASSERT(itr != cacheMapBySql.end(),
                    "A render query result cache for sql \"" + sqlStr + "\" does not exist.");

  auto numBytes = updatedResults->getQueryMemDesc().getBufferSizeBytes(updatedResults->getDeviceType());
  return _totalCachedBytes + numBytes - (*itr)->cachedBytes <= _maxTotalCachedBytes;
}

bool NonProjectionRenderQueryCacheMap::canFitUpdatedResults(const TableId tableId,
                                                            const std::shared_ptr<ResultRows>& updatedResults) {
  _purgeUnusedCaches();
  auto itr = _queryResultMap.find(tableId);

  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id: " + std::to_string(tableId) + " does not exist.");

  auto numBytes = updatedResults->getQueryMemDesc().getBufferSizeBytes(updatedResults->getDeviceType());
  return _totalCachedBytes + numBytes - (*itr)->cachedBytes <= _maxTotalCachedBytes;
}

std::pair<const ResultRows*, const std::vector<TargetMetaInfo>*> NonProjectionRenderQueryCacheMap::getQueryCacheResults(
    const TableId tableId) const {
  auto itr = _queryResultMap.find(tableId);
  RUNTIME_EX_ASSERT(itr != _queryResultMap.end(),
                    "A render query result cache for table id " + std::to_string(tableId) + " does not exist.");
  return std::make_pair((*itr)->results.get(), &(*itr)->resultsRowShape);
}

bool RootCache::supportsInt64() const {
  if (!_supportsInt64Ptr) {
    _supportsInt64Ptr.reset(
        new bool(supportedExtensions.find("GL_NV_vertex_attrib_integer_64bit") != supportedExtensions.end()));
  }

  return *_supportsInt64Ptr;
}

DistributedRenderBufferCompositorShPtr RootCache::getDistributedCompositorPtr() {
  RUNTIME_EX_ASSERT(perGpuData && perGpuData->size(),
                    "Cannot get a distributed render compositor. There are no render contexts initialized.");
  return (*perGpuData->begin())->getDistributedCompositorPtr(supportsInt64());
}

}  // namespace QueryRenderer
