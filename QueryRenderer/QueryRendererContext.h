#ifndef QUERYRENDERER_QUERYRENDERERCONTEXT_H_
#define QUERYRENDERER_QUERYRENDERERCONTEXT_H_

#include "Types.h"
#include "QueryRenderer.h"
#include "Marks/Types.h"
#include "Data/Types.h"
#include "Scales/Types.h"

#include "Utils/RapidJSONUtils.h"

class Executor;

namespace QueryRenderer {

class QueryRendererContext {
 public:
  typedef std::shared_ptr<BaseScale> ScaleShPtr;
  typedef std::function<void(RefEventType, const ScaleShPtr&)> RefEventCallback;

  explicit QueryRendererContext(int userId,
                                int widgetId,
                                const RootCacheShPtr& qrmGpuCache,
                                bool doHitTest = false,
                                bool doDepthTest = false);

  explicit QueryRendererContext(int userId,
                                int widgetId,
                                const RootCacheShPtr& qrmGpuCache,
                                int width,
                                int height,
                                bool doHitTest = false,
                                bool doDepthTest = false);

  ~QueryRendererContext();

  size_t getWidth() { return _width; }
  size_t getHeight() { return _height; }

  int getUserId() const { return _userWidget.userId; }
  int getWidgetId() const { return _userWidget.widgetId; }
  const UserWidgetIdPair& getUserWidgetIds() const { return _userWidget; }

  bool doHitTest() const { return _doHitTest; }
  bool doDepthTest() const { return _doDepthTest; }
  size_t numSamples() const {
    auto qrmGpuCache = _qrmGpuCache.lock();
    CHECK(qrmGpuCache);
    return qrmGpuCache->numSamples;
  }

  const Executor* const getExecutor() { return executor_; }
  const RapidJSONUtils::JsonCachePtr& getJsonCachePtr() { return _jsonCache; }

  const RootCacheShPtr getRootGpuCache() const { return _qrmGpuCache.lock(); }

  bool hasDataTable(const std::string& tableName) const;
  QueryDataTableShPtr getDataTable(const std::string& tableName) const;

  bool hasScale(const std::string& scaleConfigName) const;
  ScaleShPtr getScale(const std::string& scaleConfigName) const;

  bool isJSONCacheUpToDate(const rapidjson::Pointer& objPath, const rapidjson::Value& obj);

  // bool hasMark(const std::string& geomConfigName) const {
  //     return (_geomConfigMap.find(geomConfigName) != _geomConfigMap.end());
  // }

  // GeomConfigShPtr getMark(const std::string& geomConfigName) {
  //     GeomConfigShPtr rtn(nullptr);

  //     auto itr = _geomConfigMap.find(geomConfigName);
  //     if (itr != _geomConfigMap.end()) {
  //         rtn = itr->second;
  //     }

  //     return rtn;
  // }

  void subscribeToRefEvent(RefEventType eventType, const ScaleShPtr& eventObj, RefEventCallback cb);
  void unsubscribeFromRefEvent(RefEventType eventType, const ScaleShPtr& eventObj, RefEventCallback cb);

  std::set<GpuId> getUsedGpus() const;

  friend class QueryRenderer;

 private:
  typedef std::unordered_map<std::string, ScaleShPtr> ScaleConfigMap;

  typedef std::shared_ptr<BaseMark> GeomConfigShPtr;

  typedef std::vector<GeomConfigShPtr> GeomConfigVector;
  typedef std::unordered_map<std::string, QueryDataTableShPtr> DataTableMap;

  struct func_compare {
    bool operator()(const RefEventCallback& lhs, const RefEventCallback& rhs) const {
      return lhs.target_type().hash_code() < rhs.target_type().hash_code();
    }
  };
  typedef std::set<RefEventCallback, func_compare> EventCallbackList;
  typedef std::array<EventCallbackList, static_cast<size_t>(RefEventType::ALL)> EventCallbacksArray;
  typedef std::unordered_map<std::string, EventCallbacksArray> EventCallbacksMap;

  std::weak_ptr<RootCache> _qrmGpuCache;
  DataTableMap _dataTableMap;
  ScaleConfigMap _scaleConfigMap;
  GeomConfigVector _geomConfigs;

  const Executor* executor_;

  UserWidgetIdPair _userWidget;
  size_t _width;
  size_t _height;
  bool _doHitTest;
  bool _doDepthTest;

  RapidJSONUtils::JsonCachePtr _jsonCache;

  EventCallbacksMap _eventCallbacksMap;

  QueryDataLayoutShPtr _queryDataLayoutPtr;

  void _clear(bool preserveDimensions = false);
  // void _clearGpuResources();
  // void _initGpuResources(QueryRenderer::PerGpuDataMap& qrPerGpuData, const std::unordered_set<GpuId>& unusedGpus);
  void _updateConfigGpuResources();

  void _fireRefEvent(RefEventType eventType, const ScaleShPtr& eventObj);

  void _update();
};

};  // QueryRenderer namespace

#endif  // QUERYRENDERER_QUERYRENDERERCONTEXT_H_
