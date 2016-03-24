#ifndef QUERYRENDERER_QUERYRENDERER_H_
#define QUERYRENDERER_QUERYRENDERER_H_

#include "Types.h"
#include "QueryRenderManager.h"
#include "QueryRenderCompositor.h"
#include "QueryFramebuffer.h"
#include "QueryIdMapPixelBuffer.h"
#include "RapidJSONUtils.h"

#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Objects/Array2d.h>

#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>

#ifdef HAVE_CUDA
#include <CudaMgr/CudaMgr.h>
#endif  // HAVE_CUDA

class Executor;

namespace QueryRenderer {

class QueryRenderer {
 public:
  struct PerGpuData {
    QueryRenderManager::PerGpuDataWkPtr qrmGpuData;

    PerGpuData() : qrmGpuData() {}

    PerGpuData(PerGpuData&& data) noexcept : qrmGpuData(std::move(data.qrmGpuData)) {}

    ~PerGpuData() {
      // need to make active to properly destroy gpu resources
      // TODO(croot): uncomment this if we have GL resources at
      // this level (i.e. a framebuffer or a compositor per gpu)
      // TODO(croot): reset to previously active renderer?
      // makeActiveOnCurrentThread();
    }

    QueryRenderManager::PerGpuDataShPtr getQRMGpuData() const { return qrmGpuData.lock(); }

    void makeActiveOnCurrentThread() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        qrmGpuDataShPtr->makeActiveOnCurrentThread();
      }
    }

    void makeInactive() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        qrmGpuDataShPtr->makeInactive();
      }
    }

    ::Rendering::Renderer* getRenderer() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        return qrmGpuDataShPtr->getRenderer();
      }

      return nullptr;
    }

    ::Rendering::GL::GLRenderer* getGLRenderer() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        return qrmGpuDataShPtr->getGLRenderer();
      }

      return nullptr;
    }

    void resize(size_t width, size_t height) {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        qrmGpuDataShPtr->resize(width, height);
      }
    }

    QueryFramebufferUqPtr& getFramebuffer() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      CHECK(qrmGpuDataShPtr);
      return qrmGpuDataShPtr->getFramebuffer();
    }

    std::shared_ptr<QueryRenderCompositor>& getCompositor() {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      CHECK(qrmGpuDataShPtr);
      return qrmGpuDataShPtr->getCompositor();
    }

    GpuId getCompositorGpuId() {
      auto compositorPtr = getCompositor();
      ::Rendering::GL::GLRenderer* renderer = compositorPtr->getGLRenderer();
      CHECK(renderer);
      return renderer->getGpuId();
    }

    QueryIdMapPixelBufferShPtr getInactiveIdMapPbo(size_t width, size_t height) {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      CHECK(qrmGpuDataShPtr);
      return qrmGpuDataShPtr->getInactiveIdMapPbo(width, height);
    }

    void setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo) {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        qrmGpuDataShPtr->setIdMapPboInactive(pbo);
      }
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  explicit QueryRenderer(const std::shared_ptr<QueryRenderManager::PerGpuDataMap>& qrmPerGpuData,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  explicit QueryRenderer(const std::shared_ptr<QueryRenderManager::PerGpuDataMap>& qrmPerGpuData,
                         const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  explicit QueryRenderer(const std::shared_ptr<QueryRenderManager::PerGpuDataMap>& qrmPerGpuData,
                         const std::string& configJSON,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  ~QueryRenderer();

  size_t getWidth();
  size_t getHeight();
  void setWidthHeight(size_t width, size_t height);

  void setJSONConfig(const std::string& configJSON, bool forceUpdate = false);
  void setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);

#ifdef HAVE_CUDA
  void updateResultsPostQuery(QueryDataLayoutShPtr& dataLayoutPtr, const Executor* executor);
#endif  // HAVE_CUDA

  void activateGpus(const std::vector<GpuId>& gpusToActivate = {});

  void render(bool inactivateRendererOnThread = true);
  PngData renderToPng(int compressionLevel = -1);

  unsigned int getIdAt(size_t x, size_t y, size_t pixelRadius = 0);

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
  std::weak_ptr<QueryRenderManager::PerGpuDataMap> _qrmPerGpuData;
  std::shared_ptr<QueryRendererContext> _ctx;
  PerGpuDataMap _perGpuData;

  GpuId _pboGpu;
  QueryIdMapPixelBufferShPtr _pbo;

  typedef ::Rendering::Objects::Array2d<unsigned int> Array2dui;
  bool _idPixelsDirty;
  std::shared_ptr<Array2dui> _idPixels;

  void _clear();
  void _clearGpuResources();
  std::unordered_set<GpuId> _initUnusedGpus();
  void _initGpuResources(QueryRenderManager::PerGpuDataMap* qrmPerGpuData,
                         const std::vector<GpuId>& gpuIds,
                         std::unordered_set<GpuId>& unusedGpus);
  void _updateGpuData(const GpuId& gpuId,
                      QueryRenderManager::PerGpuDataMap* qrmPerGpuData,
                      std::unordered_set<GpuId>& unusedGpus,
                      size_t width,
                      size_t height);
  void _initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);
  void _initFromJSON(const std::string& configJSON, bool forceUpdate = false);
  void _resizeFramebuffers(int width, int height);

  void _update();
  void _createPbo(int width = -1, int height = -1, bool makeContextInactive = false);
  void _releasePbo(bool makeContextInactive = false);

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

    void makeActiveOnCurrentThread() const {
      QueryRenderManager::PerGpuDataShPtr qrmGpuDataShPtr = qrmGpuData.lock();
      if (qrmGpuDataShPtr) {
        qrmGpuDataShPtr->makeActiveOnCurrentThread();
      }
    }
  };
  typedef std::map<GpuId, PerGpuData> PerGpuDataMap;

  typedef std::shared_ptr<BaseScale> ScaleShPtr;
  typedef std::function<void(RefEventType, const ScaleShPtr&)> RefEventCallback;

  explicit QueryRendererContext(bool doHitTest = false, bool doDepthTest = false);
  explicit QueryRendererContext(int width, int height, bool doHitTest = false, bool doDepthTest = false);

  ~QueryRendererContext();

  size_t getWidth() { return _width; }
  size_t getHeight() { return _height; }

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

  size_t _width;
  size_t _height;
  bool _doHitTest;
  bool _doDepthTest;
  int64_t _invalidKey;

  RapidJSONUtils::JsonCachePtr _jsonCache;

  EventCallbacksMap _eventCallbacksMap;

  QueryDataLayoutShPtr _queryDataLayoutPtr;

  void _clear();
  void _clearGpuResources();
  void _initGpuResources(QueryRenderer::PerGpuDataMap& qrPerGpuData, const std::unordered_set<GpuId>& unusedGpus);
  void _updateConfigGpuResources(const std::unordered_set<GpuId>& unusedGpus);

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
