#ifndef QUERYRENDERER_QUERYRENDERERCONTEXT_H_
#define QUERYRENDERER_QUERYRENDERERCONTEXT_H_

#include "Types.h"
#include "QueryRenderer.h"
#include "Marks/Types.h"
#include "Data/Types.h"
#include "Scales/Types.h"
#include "Utils/RapidJSONUtils.h"

namespace QueryRenderer {

class QueryRendererContext {
 public:
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

  size_t getWidth() const { return _width; }
  size_t getHeight() const { return _height; }

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

  Executor* const getExecutor() { return executor_; }
  QueryExecCB getQueryExecutionFunc() { return execFunc_; }
  std::shared_ptr<RenderQueryExecuteTimer> getRenderTimer() const { return renderTimer_; }

  const RapidJSONUtils::JsonCachePtr& getJsonCachePtr() { return _jsonCache; }

  RootCacheShPtr getRootGpuCache() { return _qrmGpuCache.lock(); }
  const RootCacheShPtr getRootGpuCache() const { return _qrmGpuCache.lock(); }

  bool hasDataTable(const std::string& tableName) const;
  QueryDataTableShPtr getDataTable(const std::string& tableName) const;

  bool hasScale(const std::string& scaleConfigName) const;
  ScaleShPtr getScale(const std::string& scaleConfigName) const;

  bool isJSONCacheUpToDate(const rapidjson::Pointer& objPath, const rapidjson::Value& obj);
  const rapidjson::Value* getJSONObj(const rapidjson::Pointer& objPath);

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

  RefCallbackId subscribeToRefEvent(RefEventType eventType, const RefObjShPtr& eventObj, RefEventCallback cb);
  void unsubscribeFromRefEvent(RefEventType eventType, const RefObjShPtr& eventObj, const RefCallbackId cbId);

  std::set<GpuId> getUsedGpus() const;

  void addAccumulatorScale(const std::string& scaleName);
  void removeAccumulatorScale(const std::string& scaleName);

  friend class QueryRenderer;

 private:
  struct ScaleName {};
  typedef boost::multi_index_container<
      ScaleShPtr,
      boost::multi_index::indexed_by<
          boost::multi_index::random_access<>,

          // hashed on name
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<ScaleName>,
              boost::multi_index::const_mem_fun<JSONRefObject, std::string, &JSONRefObject::getName>>>> ScaleConfigMap;
  typedef ScaleConfigMap::index<ScaleName>::type ScaleConfigMap_by_name;

  struct DataTableName {};
  typedef boost::multi_index_container<
      QueryDataTableJSONShPtr,
      boost::multi_index::indexed_by<
          boost::multi_index::random_access<>,

          // hashed on name
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<DataTableName>,
              boost::multi_index::const_mem_fun<JSONRefObject, std::string, &JSONRefObject::getName>>>> DataTableMap;
  typedef DataTableMap::index<DataTableName>::type DataTableMap_by_name;

  typedef std::shared_ptr<BaseMark> GeomConfigShPtr;
  typedef std::vector<GeomConfigShPtr> GeomConfigVector;

  typedef std::unordered_map<RefCallbackId, RefEventCallback> EventCallbackMap;
  typedef std::array<EventCallbackMap, static_cast<size_t>(RefEventType::ALL)> EventCallbacksArray;

  typedef std::unordered_map<std::string, EventCallbacksArray> EventCallbacksByNameMap;
  typedef std::unordered_map<int, EventCallbacksByNameMap> EventCallbacksMap;

  std::weak_ptr<RootCache> _qrmGpuCache;
  DataTableMap _dataTableMap;
  ScaleConfigMap _scaleConfigMap;
  GeomConfigVector _geomConfigs;

  std::unordered_set<std::string> _accumulatorScales;

  Executor* executor_;
  QueryExecCB execFunc_;
  std::shared_ptr<RenderQueryExecuteTimer> renderTimer_;

  UserWidgetIdPair _userWidget;
  size_t _width;
  size_t _height;
  bool _doHitTest;
  bool _doDepthTest;

  RapidJSONUtils::JsonCachePtr _jsonCache;

  RefCallbackId _currCbId;
  EventCallbacksMap _eventCallbacksMap;

  QueryDataLayoutShPtr _queryDataLayoutPtr;

  void _clear(bool preserveDimensions = false);
  void _updateConfigGpuResources();

  void _fireRefEvent(RefEventType eventType, const RefObjShPtr& eventObj);

  void _update();
};
};  // QueryRenderer namespace

#endif  // QUERYRENDERER_QUERYRENDERERCONTEXT_H_
