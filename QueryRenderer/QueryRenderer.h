#ifndef QUERY_RENDERER_H_
#define QUERY_RENDERER_H_

#include "QueryRendererError.h"
#include "QueryDataLayout.h"
#include "QueryResultVertexBuffer.h"
#include "QueryFramebuffer.h"
#include "QueryRendererObjects.h"
#include "QueryFramebuffer.h"
// #include "QueryRenderManager.h"
#include "DataTable.h"
#include "BufferLayout.h"
#include "RapidJSONUtils.h"

#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <memory>  // std::unique_ptr
#include <cstdint>
#include <limits>

#include "rapidjson/document.h"
#include "rapidjson/pointer.h"
// #include <utility>  // std::pair

class Executor;

namespace MapD_Renderer {

class QueryRendererContext;
class BaseMark;
class BaseScale;

class QueryRenderer {
 public:
  explicit QueryRenderer(const Executor* executor,
                         const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false,
                         GLFWwindow* win = nullptr);

  explicit QueryRenderer(const Executor* executor,
                         const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                         const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false,
                         GLFWwindow* win = nullptr);

  explicit QueryRenderer(const Executor* executor,
                         const std::string& configJSON,
                         const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false,
                         GLFWwindow* win = nullptr);

  ~QueryRenderer();

  int getWidth();
  int getHeight();
  void setWidthHeight(int width, int height, GLFWwindow* win = nullptr);

  const QueryFramebufferUqPtr& getFramebuffer();

  void setJSONConfig(const std::string& configJSON, bool forceUpdate = false, GLFWwindow* win = nullptr);
  void setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                       bool forceUpdate = false,
                       GLFWwindow* win = nullptr);

  // void updateQueryResultBufferPostQuery(const BufferLayoutShPtr& layoutPtr,
  //                                       const int numRows,
  //                                       const int64_t invalid_key);

  void updateQueryResultBufferPostQuery(QueryDataLayout* dataLayoutPtr);

  void render();

  unsigned int getIdAt(int x, int y);

 private:
  std::shared_ptr<QueryRendererContext> _ctx;
  QueryFramebufferUqPtr _framebufferPtr;

  void _clear();
  void _initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                     bool forceUpdate = false,
                     GLFWwindow* win = nullptr);
  void _initFromJSON(const std::string& configJSON, bool forceUpdate = false, GLFWwindow* win = nullptr);
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

  explicit QueryRendererContext(const Executor* executor,
                                const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                                bool doHitTest = false,
                                bool doDepthTest = false)
      : executor_(executor),
        _queryResultVBOPtr(queryResultVBOPtr),
        _width(0),
        _height(0),
        _doHitTest(doHitTest),
        _doDepthTest(doDepthTest),
        _invalidKey(std::numeric_limits<int64_t>::max()),
        _jsonCache(nullptr) {}
  explicit QueryRendererContext(const Executor* executor,
                                const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                                int width,
                                int height,
                                bool doHitTest = false,
                                bool doDepthTest = false)
      : executor_(executor),
        _queryResultVBOPtr(queryResultVBOPtr),
        _width(width),
        _height(height),
        _doHitTest(doHitTest),
        _doDepthTest(doDepthTest),
        _invalidKey(std::numeric_limits<int64_t>::max()),
        _jsonCache(nullptr),
        _queryDataLayoutPtr(nullptr) {}
  ~QueryRendererContext() { _clear(); }

  int getWidth() { return _width; }

  int getHeight() { return _height; }

  bool hasDataTable(const std::string& tableName) const;
  DataVBOShPtr getDataTable(const std::string& tableName) const;

  bool hasScale(const std::string& scaleConfigName) const;
  ScaleShPtr getScale(const std::string& scaleConfigName) const;

  QueryResultVertexBufferShPtr getQueryResultVertexBuffer() const { return _queryResultVBOPtr; }

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
  typedef std::unordered_map<std::string, DataVBOShPtr> DataTableMap;

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
  QueryResultVertexBufferShPtr _queryResultVBOPtr;
  BufferLayoutShPtr _queryResultBufferLayout;

  int _width;
  int _height;
  bool _doHitTest;
  bool _doDepthTest;
  int64_t _invalidKey;

  RapidJSONUtils::JsonCachePtr _jsonCache;

  EventCallbacksMap _eventCallbacksMap;

  std::unique_ptr<QueryDataLayout> _queryDataLayoutPtr;

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
BaseDataTableVBO::DataTableType getDataTableTypeFromJSONObj(const rapidjson::Value& obj);
DataVBOShPtr createDataTable(const rapidjson::Value& obj,
                             const rapidjson::Pointer& objPath,
                             const QueryRendererContextShPtr& ctx,
                             const std::string& name = "");

};  // MapD_Renderer namespace

#endif  // QUERY_RENDERER_H_
