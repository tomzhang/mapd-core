#ifndef QUERYRENDERER_QUERYRENDERMANAGER_H_
#define QUERYRENDERER_QUERYRENDERMANAGER_H_

#include "Types.h"
#include "QueryDataLayout.h"
#include "QueryResultVertexBuffer.h"
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
    QueryResultVertexBufferShPtr queryResultBufferPtr;
    Rendering::WindowShPtr windowPtr;
    Rendering::RendererShPtr rendererPtr;
    // QueryRendererFboShPtr rendererFboPtr;

    PerGpuData() : queryResultBufferPtr(nullptr), windowPtr(nullptr), rendererPtr(nullptr) {}
    PerGpuData(const PerGpuData& data)
        : queryResultBufferPtr(data.queryResultBufferPtr), windowPtr(data.windowPtr), rendererPtr(data.rendererPtr) {}

    ~PerGpuData() {
      // need to make active to properly destroy gpu resources
      // TODO(croot): uncomment this if we have GL resources at
      // this level (i.e. a framebuffer or a compositor per gpu)
      // TODO(croot): reset to previously active renderer?
      // makeActiveOnCurrentThread();
    }

    void makeActiveOnCurrentThread() {
      CHECK(windowPtr && rendererPtr);
      rendererPtr->makeActiveOnCurrentThread(windowPtr);
    }

    void makeInactive() {
      CHECK(rendererPtr);
      rendererPtr->makeInactive();
    }

    Rendering::Renderer* getRenderer() {
      if (rendererPtr) {
        return rendererPtr.get();
      }
      return nullptr;
    }
  };

  typedef std::shared_ptr<PerGpuData> PerGpuDataShPtr;
  typedef std::weak_ptr<PerGpuData> PerGpuDataWkPtr;
  typedef std::map<GpuId, PerGpuDataShPtr> PerGpuDataMap;

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

  std::vector<GpuId> getAllGpuIds() const;

#ifdef HAVE_CUDA
  CudaHandle getCudaHandle(const GpuId& gpuId);
  void setCudaHandleUsedBytes(GpuId gpuId, size_t numUsedBytes);
  void configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                       QueryDataLayoutShPtr dataLayoutPtr = nullptr,
                       const Executor* executor = nullptr);
#else
  void configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr);

#endif  // HAVE_CUDA

  void render();
  PngData renderToPng(int compressionLevel = -1);

  // get the id at a specific pixel
  int64_t getIdAt(size_t x, size_t y);

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

  RendererMap _rendererMap;

  mutable RendererMap::iterator _activeItr;

  PerGpuDataMap _perGpuData;

  void _initialize(Rendering::WindowManager& windowMgr, int numGpus, int startGpu, size_t queryResultBufferSize);
  void _resetQueryResultBuffers();

  void _setActiveUserWidget(int userId, int widgetId) const;
  QueryRenderer* _getRendererForUserWidget(int userId, int widgetId) const;

  static const std::chrono::milliseconds maxWidgetIdleTime;
  void _clearActiveUserWidget();
  void _purgeUnusedWidgets();
  void _updateActiveLastRenderTime();

  mutable std::mutex _renderMtx;
  mutable std::mutex _usersMtx;

  const size_t _renderCacheLimit;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRENDERMANAGER_H_
