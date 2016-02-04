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
#include <vector>
#include <unordered_map>
#include <set>
// #include <cstdint>
// #include <limits>

// #include "rapidjson/document.h"
// #include "rapidjson/pointer.h"
// // #include <utility>  // std::pair

class Executor;

namespace QueryRenderer {

class QueryRenderer {
 public:
  explicit QueryRenderer(std::vector<QueryRenderManager::PerGpuData>& perGpuData,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  explicit QueryRenderer(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                         std::vector<QueryRenderManager::PerGpuData>& perGpuData,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  explicit QueryRenderer(const std::string& configJSON,
                         std::vector<QueryRenderManager::PerGpuData>& perGpuData,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  ~QueryRenderer();

  int getWidth();
  int getHeight();
  void setWidthHeight(int width, int height);

  const QueryFramebufferUqPtr& getFramebuffer(size_t gpuId = 0);

  void setJSONConfig(const std::string& configJSON, bool forceUpdate = false);
  void setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);

  // void updateQueryResultBufferPostQuery(const BufferLayoutShPtr& layoutPtr,
  //                                       const int numRows,
  //                                       const int64_t invalid_key);

  void updateQueryResultBufferPostQuery(QueryDataLayout* dataLayoutPtr);

  void render();

  unsigned int getIdAt(int x, int y);

 private:
  struct PerGpuData {
    QueryRenderManager::PerGpuData* qrmGpuData;
    QueryFramebufferUqPtr framebufferPtr;

    PerGpuData() : qrmGpuData(nullptr), framebufferPtr(nullptr) {}
  };

  std::shared_ptr<QueryRendererContext> _ctx;
  std::vector<PerGpuData> _perGpuData;

  void _clear();
  void _initialize(std::vector<QueryRenderManager::PerGpuData>& qrmPerGpuData);
  void _initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);
  void _initFromJSON(const std::string& configJSON, bool forceUpdate = false);
  void _initFramebuffer(int width, int height);
};

typedef std::unique_ptr<QueryRenderer> QueryRendererUqPtr;
typedef std::shared_ptr<QueryRenderer> QueryRendererShPtr;

enum class RefEventType { UPDATE = 0, REMOVE, REPLACE, ALL };

class QueryRendererContext {
 public:
  typedef std::shared_ptr<BaseScale> ScaleShPtr;
  typedef std::function<void(RefEventType, const ScaleShPtr&)> RefEventCallback;
  // typedef std::function<void(int)> RefEventCallback;

  explicit QueryRendererContext(std::vector<QueryRenderManager::PerGpuData>& perGpuData,
                                bool doHitTest = false,
                                bool doDepthTest = false)
      : executor_(nullptr),
        _perGpuData(perGpuData.size()),
        _width(0),
        _height(0),
        _doHitTest(doHitTest),
        _doDepthTest(doDepthTest),
        _invalidKey(std::numeric_limits<int64_t>::max()),
        _jsonCache(nullptr) {
    _initialize(perGpuData);
  }

  explicit QueryRendererContext(std::vector<QueryRenderManager::PerGpuData>& perGpuData,
                                int width,
                                int height,
                                bool doHitTest = false,
                                bool doDepthTest = false)
      : executor_(nullptr),
        _perGpuData(perGpuData.size()),
        _width(width),
        _height(height),
        _doHitTest(doHitTest),
        _doDepthTest(doDepthTest),
        _invalidKey(std::numeric_limits<int64_t>::max()),
        _jsonCache(nullptr),
        _queryDataLayoutPtr(nullptr) {
    _initialize(perGpuData);
  }
  ~QueryRendererContext() { _clear(); }

  int getWidth() { return _width; }

  int getHeight() { return _height; }

  bool hasDataTable(const std::string& tableName) const;
  QueryDataTableVBOShPtr getDataTable(const std::string& tableName) const;

  bool hasScale(const std::string& scaleConfigName) const;
  ScaleShPtr getScale(const std::string& scaleConfigName) const;

  QueryResultVertexBufferShPtr getQueryResultVertexBuffer(size_t gpuId = 0) const {
    RUNTIME_EX_ASSERT(gpuId < _perGpuData.size(),
                      "Invalid gpuId " + std::to_string(gpuId) + ". There are only " +
                          std::to_string(_perGpuData.size()) + " GPUs available.");
    return _perGpuData[gpuId].qrmGpuData->queryResultBufferPtr;
  }

  bool doHitTest() { return _doHitTest; }
  bool doDepthTest() { return _doDepthTest; }

  int64_t getInvalidKey() { return _invalidKey; }

  const Executor* const getExecutor() { return executor_; }

  const RapidJSONUtils::JsonCachePtr& getJsonCachePtr() { return _jsonCache; }

  bool isJSONCacheUpToDate(const rapidjson::Pointer& objPath, const rapidjson::Value& obj) {
    if (!_jsonCache) {
      return false;
    }

    const rapidjson::Value* cachedVal = GetValueByPointer(*_jsonCache, objPath);

    // TODO(croot): throw an exception or just return false?
    RUNTIME_EX_ASSERT(
        cachedVal != nullptr,
        "The path " + RapidJSONUtils::getPointerPath(objPath) + " is not a valid path in the cached json.");

    return (cachedVal ? (*cachedVal == obj) : false);
  }

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

  struct PerGpuData {
    QueryRenderManager::PerGpuData* qrmGpuData;

    PerGpuData() : qrmGpuData(nullptr) {}
  };
  std::vector<PerGpuData> _perGpuData;
  ::Rendering::GL::Resources::GLBufferLayoutShPtr _queryResultBufferLayout;

  int _width;
  int _height;
  bool _doHitTest;
  bool _doDepthTest;
  int64_t _invalidKey;

  RapidJSONUtils::JsonCachePtr _jsonCache;

  EventCallbacksMap _eventCallbacksMap;

  std::unique_ptr<QueryDataLayout> _queryDataLayoutPtr;

  void _initialize(std::vector<QueryRenderManager::PerGpuData>& qrmPerGpuData);

  void _clear() {
    _width = 0;
    _height = 0;
    _dataTableMap.clear();
    _scaleConfigMap.clear();
    _geomConfigs.clear();
    _eventCallbacksMap.clear();
  }

  void _fireRefEvent(RefEventType eventType, const ScaleShPtr& eventObj);
};

typedef std::unique_ptr<QueryRendererContext> QueryRendererContextUqPtr;
typedef std::shared_ptr<QueryRendererContext> QueryRendererContextShPtr;

std::string getDataTableNameFromJSONObj(const rapidjson::Value& obj);
QueryDataTableType getDataTableTypeFromJSONObj(const rapidjson::Value& obj);
QueryDataTableVBOShPtr createDataTable(const rapidjson::Value& obj,
                                       const rapidjson::Pointer& objPath,
                                       const QueryRendererContextShPtr& ctx,
                                       const std::string& name = "");

};  // QueryRenderer namespace

#endif  // QUERYRENDERER_QUERYRENDERER_H_
