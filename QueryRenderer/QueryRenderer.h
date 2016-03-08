#ifndef QUERYRENDERER_QUERYRENDERER_H_
#define QUERYRENDERER_QUERYRENDERER_H_

#include "Types.h"
#include "QueryRenderManager.h"
#include "QueryRenderCompositor.h"

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

#ifdef HAVE_CUDA
#include <CudaMgr/CudaMgr.h>
#endif  // HAVE_CUDA

// #include <cstdint>
// #include <limits>

// #include "rapidjson/document.h"
// #include "rapidjson/pointer.h"
// // #include <utility>  // std::pair

class Executor;

namespace QueryRenderer {

class QueryRenderer {
 public:
  struct PerGpuData {
    QueryRenderManager::PerGpuDataWkPtr qrmGpuData;
    QueryFramebufferUqPtr framebufferPtr;

    PerGpuData() : qrmGpuData(), framebufferPtr(nullptr) {}
    PerGpuData(PerGpuData&& data) noexcept : qrmGpuData(std::move(data.qrmGpuData)),
                                             framebufferPtr(std::move(data.framebufferPtr)) {}
    ~PerGpuData() {
      // need to make active to properly destroy gpu resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }

    QueryRenderManager::PerGpuDataShPtr getQRMGpuData() const { return qrmGpuData.lock(); }

    void makeActiveOnCurrentThread() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        qrmGpuDataShPtr->makeActiveOnCurrentThread();
      }
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

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

#ifdef HAVE_CUDA
  void updateQueryResultBufferPostQuery(QueryDataLayoutShPtr& dataLayoutPtr,
                                        const ::CudaMgr_Namespace::CudaMgr* cudaMgr,
                                        const Executor* executor,
                                        QueryRenderManager::PerGpuDataMap& qrmPerGpuData);
#endif  // HAVE_CUDA

  void activateGpus(QueryRenderManager::PerGpuDataMap& qrmPerGpuData, const std::vector<GpuId>& gpusToActivate = {});

  void render();
  PngData renderToPng(int compressionLevel = -1);

  unsigned int getIdAt(int x, int y);

  PerGpuDataMap* getPerGpuData() { return &_perGpuData; }
  QueryRendererContext* getContext() { return _ctx.get(); }

  static void renderGpu(GpuId gpuId,
                        PerGpuDataMap* gpuDataMap,
                        QueryRendererContext* ctx,
                        int r = -1,
                        int g = -1,
                        int b = -1,
                        int a = -1);

 private:
  std::shared_ptr<QueryRendererContext> _ctx;
  PerGpuDataMap _perGpuData;
  std::unique_ptr<QueryRenderCompositor> _compositorPtr;

  void _clear();
  void _clearGpuResources();
  void _initGpuResources(QueryRenderManager::PerGpuDataMap& qrmPerGpuData, const std::vector<GpuId>& gpuIds);
  void _updateGpuData(const GpuId& gpuId,
                      QueryRenderManager::PerGpuDataMap& qrmPerGpuData,
                      std::unordered_set<GpuId>& unusedGpus);
  void _initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);
  void _initFromJSON(const std::string& configJSON, bool forceUpdate = false);
  void _resizeFramebuffers(int width, int height);

  void _update();
  // PngData _renderToPng(int width,
  //                      int height,
  //                      const std::shared_ptr<unsigned char>& pixelsPtr,
  //                      int compressionLevel = -1);

  friend class QueryRendererContext;
  friend class QueryRenderCompositor;
};

class QueryRendererContext {
 public:
  struct PerGpuData {
    QueryRenderManager::PerGpuDataWkPtr qrmGpuData;

    PerGpuData() : qrmGpuData() {}
    PerGpuData(const QueryRenderManager::PerGpuDataWkPtr& qrmGpuData) : qrmGpuData(qrmGpuData) {}
    PerGpuData(const PerGpuData& data) : qrmGpuData(data.qrmGpuData) {}
    PerGpuData(PerGpuData&& data) : qrmGpuData(std::move(data.qrmGpuData)) {}

    QueryRenderManager::PerGpuDataShPtr getQRMGpuData() const { return qrmGpuData.lock(); }
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

  const QueryDataLayoutShPtr& getQueryDataLayout() { return _queryDataLayoutPtr; }

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

  QueryDataLayoutShPtr _queryDataLayoutPtr;

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
