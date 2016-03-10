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

#include "rapidjson/document.h"

#ifdef HAVE_CUDA
#include <CudaMgr/CudaMgr.h>
#endif  // HAVE_CUDA

class Executor;
class CudaMgr;

namespace QueryRenderer {

typedef std::pair<int, int> UserWidgetPair;
typedef std::unordered_map<int, QueryRendererUqPtr> WidgetRendererMap;
typedef std::unordered_map<int, std::unique_ptr<WidgetRendererMap>> RendererTable;

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

  explicit QueryRenderManager(int numGpus = -1, int startGpu = 0, size_t queryResultBufferSize = 500000);
  explicit QueryRenderManager(
      Rendering::WindowManager& windowMgr,
      int numGpus = -1,  // < 0 means use all available GPUs
      int startGpu = 0,
      size_t queryResultBufferSize = 500000);  // only applicable if a GPU or CUDA_INTEROP render
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

  // CROOT - the following is a debug function. Remove when ready for deployment
  // GLFWwindow* getWindow() {
  //   // return _windowPtr.get();
  //   return _windowPtr;
  // }

 private:
  static const UserWidgetPair _emptyUserWidget;

  RendererTable _rendererDict;

  mutable QueryRenderer* _activeRenderer;
  mutable UserWidgetPair _activeUserWidget;

  PerGpuDataMap _perGpuData;

  void _initialize(Rendering::WindowManager& windowMgr, int numGpus, int startGpu, size_t queryResultBufferSize);
  void _resetQueryResultBuffers();

  void _setActiveUserWidget(int userId, int widgetId) const;
  QueryRenderer* _getRendererForUserWidget(int userId, int widgetId) const;

  std::mutex _mtx;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRENDERMANAGER_H_
