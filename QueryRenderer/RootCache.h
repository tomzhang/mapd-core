#ifndef QUERYRENDERER_ROOTCACHE_H_
#define QUERYRENDERER_ROOTCACHE_H_

#include "PerGpuData.h"
#include "Data/QueryPolyDataTable.h"

namespace QueryRenderer {

struct RootCache {
  std::shared_ptr<RootPerGpuDataMap> perGpuData;
  std::unordered_map<std::string, SqlQueryPolyDataTable> polyCacheMap;

  RootCache() : perGpuData(new RootPerGpuDataMap()), polyCacheMap() {}
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_ROOTCACHE_H_
