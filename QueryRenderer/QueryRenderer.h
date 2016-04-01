#ifndef QUERYRENDERER_QUERYRENDERER_H_
#define QUERYRENDERER_QUERYRENDERER_H_

#include "Types.h"
#include "Rendering/Types.h"
#include "Interop/Types.h"
#include "Marks/Types.h"
#include "Data/Types.h"

#include "QueryRenderManager.h"
#include "Rendering/QueryRenderCompositor.h"
#include "Rendering/QueryFramebuffer.h"
#include "Rendering/QueryIdMapPixelBuffer.h"
#include "Utils/RapidJSONUtils.h"

#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Objects/Array2d.h>

#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>

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

  explicit QueryRenderer(int userId,
                         int widgetId,
                         const std::shared_ptr<QueryRenderManager::PerGpuDataMap>& qrmPerGpuData,
                         bool doHitTest = false,
                         bool doDepthTest = false) noexcept;

  explicit QueryRenderer(int userId,
                         int widgetId,
                         const std::shared_ptr<QueryRenderManager::PerGpuDataMap>& qrmPerGpuData,
                         const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  explicit QueryRenderer(int userId,
                         int widgetId,
                         const std::shared_ptr<QueryRenderManager::PerGpuDataMap>& qrmPerGpuData,
                         const std::string& configJSON,
                         bool doHitTest = false,
                         bool doDepthTest = false);

  ~QueryRenderer();

  size_t getWidth();
  size_t getHeight();
  void setWidthHeight(size_t width, size_t height);

  void setJSONConfig(const std::string& configJSON, bool forceUpdate = false);
  void setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate = false);

  void updateResultsPostQuery(QueryDataLayoutShPtr& dataLayoutPtr, const Executor* executor);
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

  void _clear(bool preserveDimensions = false);
  void _clearGpuResources();
  void _clearAll(bool preserveDimensions = false);
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

std::string getDataTableNameFromJSONObj(const rapidjson::Value& obj);
QueryDataTableType getDataTableTypeFromJSONObj(const rapidjson::Value& obj);
QueryDataTableVBOShPtr createDataTable(const rapidjson::Value& obj,
                                       const rapidjson::Pointer& objPath,
                                       const QueryRendererContextShPtr& ctx,
                                       const std::string& name = "");

};  // QueryRenderer namespace

#endif  // QUERYRENDERER_QUERYRENDERER_H_
