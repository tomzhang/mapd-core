#ifndef QUERYRENDERER_QUERYRENDERER_H_
#define QUERYRENDERER_QUERYRENDERER_H_

#include "Types.h"
#include "QueryRenderManager.h"

// #include "QueryRendererError.h"
// #include "QueryDataLayout.h"
// #include "QueryResultVertexBuffer.h"
#include "QueryFramebuffer.h"
// #include "QueryRendererObjects.h"
// #include "QueryFramebuffer.h"
// // #include "QueryRenderManager.h"
// #include "DataTable.h"
// #include "BufferLayout.h"
#include "RapidJSONUtils.h"
#include <Rendering/Renderer/GL/Resources/Types.h>

#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
// #include <cstdint>
// #include <limits>

// #include "rapidjson/document.h"
// #include "rapidjson/pointer.h"
// // #include <utility>  // std::pair

class Executor;

namespace QueryRenderer {

class QueryRenderer {
 public:
  explicit QueryRenderer(bool doHitTest = false, bool doDepthTest = false);

  explicit QueryRenderer(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  explicit QueryRenderer(const std::string& configJSON, bool doHitTest = false, bool doDepthTest = false);

  ~QueryRenderer();

  int getWidth();
  int getHeight();
  void setWidthHeight(int width, int height);

  const QueryFramebufferUqPtr& getFramebuffer(const GpuId& gpuId = 0);

  void setJSONConfig(const std::string& configJSON, bool forceUpdate = false);
  void setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);

  void updateQueryResultBufferPostQuery(QueryDataLayout* dataLayoutPtr,
                                        QueryRenderManager::PerGpuDataMap& qrmPerGpuData);

  void activateGpu(const GpuId& gpuId, QueryRenderManager::PerGpuDataMap& qrmPerGpuData);

  void render();
  PngData renderToPng(int compressionLevel = -1);

  unsigned int getIdAt(int x, int y);

 private:
  struct PerGpuData {
    QueryRenderManager::PerGpuData* qrmGpuData;
    QueryFramebufferUqPtr framebufferPtr;

    PerGpuData() : qrmGpuData(nullptr), framebufferPtr(nullptr) {}
    PerGpuData(PerGpuData&& data) noexcept : qrmGpuData(std::move(data.qrmGpuData)),
                                             framebufferPtr(std::move(data.framebufferPtr)) {}
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  std::shared_ptr<QueryRendererContext> _ctx;
  PerGpuDataMap _perGpuData;

  void _clear();
  void _clearGpuResources();
  void _initGpuResources(const std::vector<GpuId>& gpuIds, QueryRenderManager::PerGpuDataMap& qrmPerGpuData);
  void _initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);
  void _initFromJSON(const std::string& configJSON, bool forceUpdate = false);
  void _resizeFramebuffers(int width, int height);

  void _update();
  void _renderGpu(PerGpuDataMap::iterator& itr);

  friend class QueryRendererContext;
};

class QueryRendererContext {
 public:
  struct PerGpuData {
    QueryRenderManager::PerGpuData* qrmGpuData;

    PerGpuData() : qrmGpuData(nullptr) {}
    PerGpuData(QueryRenderManager::PerGpuData* qrmGpuData) : qrmGpuData(qrmGpuData) {}
    PerGpuData(const PerGpuData& data) : qrmGpuData(data.qrmGpuData) {}
    PerGpuData(PerGpuData&& data) : qrmGpuData(std::move(data.qrmGpuData)) {}
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  typedef std::shared_ptr<BaseScale> ScaleShPtr;
  typedef std::function<void(RefEventType, const ScaleShPtr&)> RefEventCallback;

  explicit QueryRendererContext(bool doHitTest = false, bool doDepthTest = false);
  explicit QueryRendererContext(int width, int height, bool doHitTest = false, bool doDepthTest = false);

  ~QueryRendererContext();

  int getWidth() { return _width; }
  int getHeight() { return _height; }

  bool doHitTest() { return _doHitTest; }
  bool doDepthTest() { return _doDepthTest; }

  int64_t getInvalidKey() { return _invalidKey; }
  const Executor* const getExecutor() { return executor_; }
  const RapidJSONUtils::JsonCachePtr& getJsonCachePtr() { return _jsonCache; }

  const PerGpuDataMap& getGpuDataMap() const { return _perGpuData; }

  bool hasDataTable(const std::string& tableName) const;
  QueryDataTableVBOShPtr getDataTable(const std::string& tableName) const;

  bool hasScale(const std::string& scaleConfigName) const;
  ScaleShPtr getScale(const std::string& scaleConfigName) const;

  QueryResultVertexBufferShPtr getQueryResultVertexBuffer(const GpuId& gpuId = 0) const;
  std::map<GpuId, QueryVertexBufferShPtr> getQueryResultVertexBuffers() const;

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

  const std::unique_ptr<QueryDataLayout>& getQueryDataLayout() { return _queryDataLayoutPtr; }

  friend class QueryRenderer;

 private:
  typedef std::unordered_map<std::string, ScaleShPtr> ScaleConfigMap;

  typedef std::shared_ptr<BaseMark> GeomConfigShPtr;

  typedef std::vector<GeomConfigShPtr> GeomConfigVector;
  typedef std::unordered_map<std::string, QueryDataTableVBOShPtr> DataTableMap;

  struct func_compare {
    bool operator()(const RefEventCallback& lhs, const RefEventCallback& rhs) const {
      return lhs.target_type().hash_code() < rhs.target_type().hash_code();
    }
  };
  typedef std::set<RefEventCallback, func_compare> EventCallbackList;
  typedef std::array<EventCallbackList, static_cast<size_t>(RefEventType::ALL)> EventCallbacksArray;
  typedef std::unordered_map<std::string, EventCallbacksArray> EventCallbacksMap;

  DataTableMap _dataTableMap;
  ScaleConfigMap _scaleConfigMap;
  GeomConfigVector _geomConfigs;

  const Executor* executor_;

  PerGpuDataMap _perGpuData;
  ::Rendering::GL::Resources::GLBufferLayoutShPtr _queryResultBufferLayout;

  int _width;
  int _height;
  bool _doHitTest;
  bool _doDepthTest;
  int64_t _invalidKey;

  RapidJSONUtils::JsonCachePtr _jsonCache;

  EventCallbacksMap _eventCallbacksMap;

  std::unique_ptr<QueryDataLayout> _queryDataLayoutPtr;

  void _clear();
  void _clearGpuResources();
  void _initGpuResources(QueryRenderer::PerGpuDataMap& qrPerGpuData, const std::unordered_set<GpuId>& unusedGpus);

  void _fireRefEvent(RefEventType eventType, const ScaleShPtr& eventObj);

  void _update();
};

std::string getDataTableNameFromJSONObj(const rapidjson::Value& obj);
QueryDataTableType getDataTableTypeFromJSONObj(const rapidjson::Value& obj);
QueryDataTableVBOShPtr createDataTable(const rapidjson::Value& obj,
                                       const rapidjson::Pointer& objPath,
                                       const QueryRendererContextShPtr& ctx,
                                       const std::string& name = "");

};  // QueryRenderer namespace

#endif  // QUERYRENDERER_QUERYRENDERER_H_
