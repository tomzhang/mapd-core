#ifndef QUERYRENDERER_QUERYRENDERMANAGER_H_
#define QUERYRENDERER_QUERYRENDERMANAGER_H_

#include "Types.h"
#include "QueryDataLayout.h"
#include "QueryResultVertexBuffer.h"
#include "QueryRenderCompositor.h"
#include "QueryIdMapPboPool.h"
#include "Types.h"
#include "PngData.h"
#include <Rendering/Types.h>

#include <unordered_map>
#include <map>
#include <utility>  // std::pair
#include <mutex>
#include <chrono>

#include "rapidjson/document.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/tag.hpp>

#ifdef HAVE_CUDA
#include <CudaMgr/CudaMgr.h>
#endif  // HAVE_CUDA

class Executor;
class CudaMgr;

namespace QueryRenderer {

typedef std::pair<int, int> UserWidgetPair;

class QueryRenderManager {
 public:
  struct PerGpuData {
    GpuId gpuId;
    QueryResultVertexBufferShPtr queryResultBufferPtr;
    Rendering::WindowShPtr windowPtr;
    Rendering::RendererShPtr rendererPtr;

    // TODO(croot): make a pool of framebuffers?
    // This would be necessary if we ever support asynchronous
    // queries, even partially (i.e. the cuda part of the query
    // may be serialized, but the render part could run asynchronously.)
    QueryFramebufferUqPtr framebufferPtr;
    std::shared_ptr<QueryRenderCompositor> compositorPtr;
    std::unique_ptr<QueryIdMapPboPool> pboPoolPtr;

    PerGpuData(GpuId gpuId)
        : gpuId(gpuId),
          queryResultBufferPtr(nullptr),
          windowPtr(nullptr),
          rendererPtr(nullptr),
          framebufferPtr(nullptr),
          compositorPtr(nullptr),
          pboPoolPtr(nullptr) {}

    ~PerGpuData() {
      // need to make active to properly destroy gpu resources
      // TODO(croot): reset to previously active renderer?
      makeActiveOnCurrentThread();
    }

    void makeActiveOnCurrentThread() {
      CHECK(windowPtr && rendererPtr);
      rendererPtr->makeActiveOnCurrentThread(windowPtr);
    }

    void makeInactive() {
      CHECK(rendererPtr);
      rendererPtr->makeInactive();
    }

    ::Rendering::Renderer* getRenderer() {
      if (rendererPtr) {
        return rendererPtr.get();
      }
      return nullptr;
    }

    ::Rendering::GL::GLRenderer* getGLRenderer() {
      if (rendererPtr) {
        ::Rendering::Renderer* renderer = rendererPtr.get();
        return dynamic_cast<::Rendering::GL::GLRenderer*>(renderer);
      }

      return nullptr;
    }

    void resize(size_t width, size_t height) {
      CHECK(framebufferPtr);

      width = std::max(width, framebufferPtr->getWidth());
      height = std::max(height, framebufferPtr->getHeight());
      framebufferPtr->resize(width, height);
      if (compositorPtr && (compositorPtr->getWidth() < width || compositorPtr->getHeight() < height)) {
        ::Rendering::GL::GLRenderer* prevRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
        ::Rendering::Window* prevWindow = ::Rendering::GL::GLRenderer::getCurrentThreadWindow();
        ::Rendering::GL::GLRenderer* renderer = compositorPtr->getGLRenderer();
        CHECK(renderer);
        bool reset = false;
        if (renderer != prevRenderer) {
          reset = true;
          renderer->makeActiveOnCurrentThread();
        }
        compositorPtr->resize(width, height);
        if (reset) {
          prevRenderer->makeActiveOnCurrentThread(prevWindow);
        }
      }
    }

    QueryFramebufferUqPtr& getFramebuffer() { return framebufferPtr; }
    std::shared_ptr<QueryRenderCompositor>& getCompositor() { return compositorPtr; }
    std::unique_ptr<QueryIdMapPboPool>& getIdMapPboPool() { return pboPoolPtr; }

    QueryIdMapPixelBufferShPtr getInactiveIdMapPbo(size_t width, size_t height) {
      CHECK(pboPoolPtr);
      return pboPoolPtr->getInactiveIdMapPbo(width, height);
    }

    void setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo) {
      CHECK(pboPoolPtr);
      pboPoolPtr->setIdMapPboInactive(pbo);
    }
  };

  typedef std::shared_ptr<PerGpuData> PerGpuDataShPtr;
  typedef std::weak_ptr<PerGpuData> PerGpuDataWkPtr;

  struct inorder {};

  struct PerGpuDataId {
    typedef GpuId result_type;

    result_type operator()(const PerGpuDataShPtr& perGpuData) const { return perGpuData->gpuId; }
  };

  typedef ::boost::multi_index_container<PerGpuDataShPtr,
                                         ::boost::multi_index::indexed_by<
                                             // hashed on gpuId
                                             ::boost::multi_index::ordered_unique<PerGpuDataId>,
                                             ::boost::multi_index::random_access<::boost::multi_index::tag<inorder>>>>
      PerGpuDataMap;

  typedef PerGpuDataMap::index<inorder>::type PerGpuDataMap_in_order;

  explicit QueryRenderManager(int numGpus = -1,
                              int startGpu = 0,
                              size_t queryResultBufferSize = 500000,
                              size_t renderCacheLimit = 500);
  explicit QueryRenderManager(Rendering::WindowManager& windowMgr,
                              int numGpus = -1,  // < 0 means use all available GPUs
                              int startGpu = 0,
                              size_t queryResultBufferSize = 500000,
                              size_t renderCacheLimit = 500);  // only applicable if a GPU or CUDA_INTEROP render
  ~QueryRenderManager();

  bool hasUser(int userId) const;
  bool hasUserWidget(int userId, int widgetId) const;
  bool hasUserWidget(const UserWidgetPair& userWidgetPair) const;

  void addUserWidget(int userId, int widgetId, bool doHitTest = false, bool doDepthTest = false);
  void addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest = false, bool doDepthTest = false);

  void removeUserWidget(int userId, int widgetId);
  void removeUserWidget(const UserWidgetPair& userWidgetPair);

  // Removes all widgets/sessions for a particular user id.
  void removeUser(int userId);

  void setActiveUserWidget(int userId, int widgetId);
  void setActiveUserWidget(const UserWidgetPair& userWidgetPair);

  void setWidthHeight(int width, int height);

  size_t getNumGpus() const;
  std::vector<GpuId> getAllGpuIds() const;
  PerGpuDataMap* getPerGpuData() { return _perGpuData.get(); }

#ifdef HAVE_CUDA
  CudaHandle getCudaHandle(size_t gpuIdx);
  void setCudaHandleUsedBytes(size_t gpuIdx, size_t numUsedBytes);
  void configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                       QueryDataLayoutShPtr dataLayoutPtr = nullptr,
                       const Executor* executor = nullptr);
#else
  void configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr);

#endif  // HAVE_CUDA

  void render();
  PngData renderToPng(int compressionLevel = -1);

  // get the id at a specific pixel
  int64_t getIdAt(size_t x, size_t y, size_t pixelRadius = 0);

 private:
  static const UserWidgetPair _emptyUserWidget;

  struct UserId {};
  struct UserWidgetIds {};
  struct LastRenderTime {};

  struct SessionData {
    int userId;
    int widgetId;
    QueryRendererUqPtr renderer;
    std::chrono::milliseconds lastRenderTime;

    SessionData(int userId, int widgetId, QueryRenderer* newRenderer);
  };

  struct ChangeLastRenderTime {
    ChangeLastRenderTime();
    void operator()(SessionData& sd);

   private:
    std::chrono::milliseconds new_time;
  };

  typedef ::boost::multi_index_container<
      SessionData,
      ::boost::multi_index::indexed_by<::boost::multi_index::hashed_unique<
                                           ::boost::multi_index::tag<UserWidgetIds>,
                                           ::boost::multi_index::composite_key<
                                               SessionData,
                                               ::boost::multi_index::member<SessionData, int, &SessionData::userId>,
                                               ::boost::multi_index::member<SessionData, int, &SessionData::widgetId>>>,

                                       ::boost::multi_index::ordered_non_unique<
                                           ::boost::multi_index::tag<UserId>,
                                           ::boost::multi_index::member<SessionData, int, &SessionData::userId>>,

                                       ::boost::multi_index::sequenced<::boost::multi_index::tag<LastRenderTime>>>>
      RendererMap;

  typedef RendererMap::index<UserId>::type RendererMap_by_UserId;
  typedef RendererMap::index<LastRenderTime>::type RendererMap_by_LastRenderTime;

  struct ActiveRendererGuard {
    QueryRenderManager::PerGpuData* currGpuData;
    QueryRenderManager* qrm;

    ActiveRendererGuard(QueryRenderManager::PerGpuData* currGpuData = nullptr, QueryRenderManager* qrm = nullptr);
    ~ActiveRendererGuard();
  };

  RendererMap _rendererMap;

  mutable RendererMap::iterator _activeItr;

  std::shared_ptr<PerGpuDataMap> _perGpuData;
  std::shared_ptr<QueryRenderCompositor> _compositorPtr;

  void _initialize(Rendering::WindowManager& windowMgr, int numGpus, int startGpu, size_t queryResultBufferSize);
  void _resetQueryResultBuffers() noexcept;

  void _setActiveUserWidget(int userId, int widgetId) const;
  QueryRenderer* _getRendererForUserWidget(int userId, int widgetId) const;

  static const std::chrono::milliseconds maxWidgetIdleTime;
  void _clearActiveUserWidget();
  void _purgeUnusedWidgets();
  void _updateActiveLastRenderTime();

  mutable std::mutex _renderMtx;

  const size_t _renderCacheLimit;

  friend struct ActiveRendererGuard;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRENDERMANAGER_H_
