#ifndef QUERYRENDERER_ROOTCACHE_H_
#define QUERYRENDERER_ROOTCACHE_H_

#include "PerGpuData.h"
#include "Data/QueryPolyDataTable.h"

namespace QueryRenderer {

struct RootCache {
  std::shared_ptr<RootPerGpuDataMap> perGpuData;
  std::unordered_map<std::string, std::pair<size_t, SqlQueryPolyDataTableCache>> polyCacheMap;
  size_t numSamples;

  RootCache(const size_t numSamples) : perGpuData(new RootPerGpuDataMap()), polyCacheMap(), numSamples(numSamples) {}

  size_t buildSqlHash(const std::string& sqlStr) const { return std::hash<std::string>{}(sqlStr); }

  bool hasPolyTableCache(const std::string& tableName) { return polyCacheMap.find(tableName) != polyCacheMap.end(); }

  bool hasPolyTableCache(const std::string& tableName, const GpuId gpuId) {
    auto itr = polyCacheMap.find(tableName);
    if (itr == polyCacheMap.end()) {
      return false;
    }

    return itr->second.second.usesGpu(gpuId);
  }

  bool hasPolyTableCache(const std::string& tableName, const std::string& sqlStr) {
    auto itr = polyCacheMap.find(tableName);
    if (itr == polyCacheMap.end()) {
      return false;
    }

    return itr->second.first == buildSqlHash(sqlStr);
  }

  bool hasPolyTableCache(const std::string& tableName, const std::string& sqlStr, const GpuId gpuId) {
    auto itr = polyCacheMap.find(tableName);
    if (itr == polyCacheMap.end()) {
      return false;
    }

    if (itr->second.first != buildSqlHash(sqlStr)) {
      return false;
    }

    return itr->second.second.usesGpu(gpuId);
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_ROOTCACHE_H_
