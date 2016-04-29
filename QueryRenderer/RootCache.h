#ifndef QUERYRENDERER_ROOTCACHE_H_
#define QUERYRENDERER_ROOTCACHE_H_

#include "PerGpuData.h"
#include "Data/QueryPolyDataTable.h"

namespace QueryRenderer {

struct RootCache {
  std::shared_ptr<RootPerGpuDataMap> perGpuData;
  std::unordered_map<std::string, SqlQueryPolyDataTable> polyCacheMap;
  size_t numSamples;

  RootCache(const size_t numSamples) : perGpuData(new RootPerGpuDataMap()), polyCacheMap(), numSamples(numSamples) {}
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_ROOTCACHE_H_
