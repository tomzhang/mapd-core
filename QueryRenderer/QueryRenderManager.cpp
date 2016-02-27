#include "QueryRenderManager.h"
#include "QueryRenderer.h"
#include <Rendering/WindowManager.h>
#include <Rendering/Settings/WindowSettings.h>
#include <Rendering/Settings/RendererSettings.h>
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/GLRenderer.h>

// #include "QueryFramebuffer.h"

// #include <GL/glew.h>
// #include <png.h>

// #include <glog/logging.h>
// #include <time.h>
// #include <iostream>
// #include <stdexcept>
// #include <fstream>
// #include <sstream>
// #include <map>
// #include "rapidjson/document.h"

namespace QueryRenderer {

using ::Rendering::Settings::WindowSettings;
using ::Rendering::Settings::RendererSettings;

using ::Rendering::Settings::IntConstant;
using ::Rendering::Settings::IntSetting;
using ::Rendering::Settings::StrSetting;

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;

const UserWidgetPair QueryRenderManager::_emptyUserWidget = std::make_pair(-1, -1);

// QueryRenderManager::QueryRenderManager(int queryResultBufferSize, bool debugMode) : _debugMode(debugMode),
// _activeRenderer(nullptr), _windowPtr(nullptr, glfwDestroyWindow), _queryResultVBOPtr(nullptr) {
QueryRenderManager::QueryRenderManager(Rendering::WindowManager& windowMgr,
                                       const Executor* executor,
                                       int numGpus,
                                       size_t queryResultBufferSize,
                                       bool debugMode)

    : _debugMode(debugMode),
      _activeRenderer(nullptr),
      _activeUserWidget(_emptyUserWidget),
      _perGpuData(),
      executor_(executor) {
  _initialize(windowMgr, numGpus, queryResultBufferSize);
}

QueryRenderManager::~QueryRenderManager() {
}

void QueryRenderManager::_initialize(Rendering::WindowManager& windowMgr, int numGpus, size_t queryResultBufferSize) {
  int maxNumGpus = windowMgr.getNumGpus();
  if (numGpus < 0) {
    numGpus = maxNumGpus;
  } else {
    RUNTIME_EX_ASSERT(numGpus <= maxNumGpus,
                      "QueryRenderManager initialization is requesting the use of " + std::to_string(numGpus) + "GPU" +
                          (maxNumGpus > 1 ? "s" : "") + " but only " + std::to_string(maxNumGpus) +
                          " are available for rendering.");
  }

  int defaultWidth = 1024, defaultHeight = 1024;  // TODO(croot): expose as a static somewhere?

  // setup the window settings
  WindowSettings windowSettings;
  windowSettings.setViewport(0, 0, defaultWidth, defaultHeight);

  // use a pbuffer as the draw type. This will essentiall create an offline window
  // that isn't dependent on a window server.
  windowSettings.setIntSetting(IntSetting::DRAWABLE_TYPE, IntConstant::PBUFFER);
  std::string windowName = "QueryRenderer on GPU ";

  // create renderer settings that will be compatible with the window's settings.
  RendererSettings rendererSettings(windowSettings);
  // rendererSettings.setIntSetting(IntSetting::OPENGL_MAJOR, 4);
  // rendererSettings.setIntSetting(IntSetting::OPENGL_MINOR, 5);

  GLRenderer* renderer;
  GLResourceManagerShPtr rsrcMgrPtr;

  PerGpuDataShPtr gpuDataPtr;
  for (int i = 0; i < numGpus; ++i) {
    windowSettings.setStrSetting(StrSetting::NAME, windowName + std::to_string(i));
    windowSettings.setIntSetting(IntSetting::GPU_ID, i);

    gpuDataPtr.reset(new PerGpuData());
    gpuDataPtr->windowPtr = windowMgr.createWindow(windowSettings);
    gpuDataPtr->rendererPtr = windowMgr.createRendererForWindow(rendererSettings, gpuDataPtr->windowPtr);

    gpuDataPtr->makeActiveOnCurrentThread();

    renderer = dynamic_cast<GLRenderer*>(gpuDataPtr->rendererPtr.get());
    CHECK(renderer != nullptr);

    gpuDataPtr->queryResultBufferPtr.reset(new QueryResultVertexBuffer(renderer, queryResultBufferSize));

    _perGpuData.insert({i, gpuDataPtr});
  }
}

void QueryRenderManager::setActiveUserWidget(int userId, int widgetId) {
  // TODO(croot): should we put thread locks in here? that probably makes sense.

  UserWidgetPair userWidget = std::make_pair(userId, widgetId);

  if (userWidget != _activeUserWidget) {
    auto userIter = _rendererDict.find(userId);

    RUNTIME_EX_ASSERT(userIter != _rendererDict.end(), "User id: " + std::to_string(userId) + " does not exist.");

    WidgetRendererMap* wfMap = userIter->second.get();

    auto widgetIter = wfMap->find(widgetId);

    RUNTIME_EX_ASSERT(
        widgetIter != wfMap->end(),
        "Widget id: " + std::to_string(widgetId) + " for user id: " + std::to_string(userId) + " does not exist.");

    _activeRenderer = widgetIter->second.get();
    _activeUserWidget = userWidget;
  }
}

void QueryRenderManager::setActiveUserWidget(const UserWidgetPair& userWidgetPair) {
  setActiveUserWidget(userWidgetPair.first, userWidgetPair.second);
}

bool QueryRenderManager::inDebugMode() const {
  return (_debugMode == true);
}

bool QueryRenderManager::hasUser(int userId) const {
  return (_rendererDict.find(userId) != _rendererDict.end());
}

bool QueryRenderManager::hasUserWidget(int userId, int widgetId) const {
  auto userIter = _rendererDict.find(userId);

  if (userIter == _rendererDict.end()) {
    return false;
  }

  return (userIter->second->find(widgetId) != userIter->second->end());
}

bool QueryRenderManager::hasUserWidget(const UserWidgetPair& userWidgetPair) const {
  return hasUserWidget(userWidgetPair.first, userWidgetPair.second);
}

void QueryRenderManager::addUserWidget(int userId, int widgetId, bool doHitTest, bool doDepthTest) {
  WidgetRendererMap* wfMap;

  auto userIter = _rendererDict.find(userId);

  if (userIter == _rendererDict.end()) {
    wfMap = new WidgetRendererMap();
    _rendererDict[userId] = std::unique_ptr<WidgetRendererMap>(wfMap);
  } else {
    wfMap = userIter->second.get();

    RUNTIME_EX_ASSERT(wfMap->find(widgetId) == wfMap->end(),
                      "Cannot add user widget. User id: " + std::to_string(userId) + " with widget id: " +
                          std::to_string(widgetId) + " already exists.");
  }

  // (*wfMap)[widgetId] = QueryRendererUqPtr(
  //     new QueryRenderer(executor_, _queryResultVBOPtr, doHitTest, doDepthTest, (_debugMode ? _windowPtr : nullptr)));
  (*wfMap)[widgetId] = QueryRendererUqPtr(new QueryRenderer(doHitTest, doDepthTest));
  // new QueryRenderer(executor_, _queryResultVBOPtr, doHitTest, doDepthTest, (_debugMode ? _windowPtr : nullptr)));

  // TODO(croot): should we set this as active the newly added ids as active?
  // setActiveUserWidget(userId, widgetId);
}

void QueryRenderManager::addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest, bool doDepthTest) {
  addUserWidget(userWidgetPair.first, userWidgetPair.second, doHitTest, doDepthTest);
}

void QueryRenderManager::removeUserWidget(int userId, int widgetId) {
  auto userIter = _rendererDict.find(userId);

  RUNTIME_EX_ASSERT(userIter != _rendererDict.end(),
                    "User id " + std::to_string(userId) + " does not exist. Cannot remove the caches for " +
                        std::to_string(userId) + ":" + std::to_string(widgetId) + ".");

  WidgetRendererMap* wfMap = userIter->second.get();
  auto widgetIter = wfMap->find(widgetId);

  RUNTIME_EX_ASSERT(widgetIter != wfMap->end(),
                    "Widget id " + std::to_string(widgetId) + " for user id " + std::to_string(userId) +
                        " does not exist. Cannot remove widget.");

  wfMap->erase(widgetIter);

  if (userId == _activeUserWidget.first && widgetId == _activeUserWidget.second) {
    _activeUserWidget = _emptyUserWidget;
    _activeRenderer = nullptr;
  }
}

void QueryRenderManager::removeUserWidget(const UserWidgetPair& userWidgetPair) {
  removeUserWidget(userWidgetPair.first, userWidgetPair.second);
}

// Removes all widgets/sessions for a particular user id.
void QueryRenderManager::removeUser(int userId) {
  auto userIter = _rendererDict.find(userId);

  RUNTIME_EX_ASSERT(userIter != _rendererDict.end(),
                    "User id " + std::to_string(userId) + " does not exist. Cannot remove its caches.");

  _rendererDict.erase(userIter);

  if (userId == _activeUserWidget.first) {
    _activeUserWidget = _emptyUserWidget;
    _activeRenderer = nullptr;
  }
}

#ifdef HAVE_CUDA
CudaHandle QueryRenderManager::getCudaHandle(const GpuId& gpuId) {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr != _perGpuData.end(), "Cannot get cuda handle for gpu " + std::to_string(gpuId) + ".");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_mtx);

  itr->second->makeActiveOnCurrentThread();
  CudaHandle rtn = itr->second->queryResultBufferPtr->getCudaHandlePreQuery();
  itr->second->rendererPtr->makeInactive();

  return rtn;
}
void QueryRenderManager::configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                                         QueryDataLayoutShPtr dataLayoutPtr,
                                         const ::CudaMgr_Namespace::CudaMgr* cudaMgr,
                                         const Executor* executor) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "ConfigureRender: There is no active user/widget id. Must set a user/widget id active before "
                    "configuring the render.");

  std::lock_guard<std::mutex> render_lock(_mtx);

  // need to update the data layout of the query result buffer before building up
  // from the json obj
  if (dataLayoutPtr) {
    // CHECK(executor != nullptr);
    CHECK(cudaMgr != nullptr);

    _activeRenderer->updateQueryResultBufferPostQuery(dataLayoutPtr, cudaMgr, executor, _perGpuData);
  } else {
    CHECK(_perGpuData.size());

    // uses the first gpu as the default.
    // TODO(croot): expose a way to specify which gpu to use?
    // auto itr = _perGpuData.begin();
    // _activeRenderer->activateGpus(_perGpuData, itr->first);
    _activeRenderer->activateGpus(_perGpuData);
  }

  _activeRenderer->setJSONDocument(jsonDocumentPtr, false);
}
#else
void QueryRenderManager::configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "ConfigureRender: There is no active user/widget id. Must set a user/widget id active before "
                    "configuring the render.");

  std::lock_guard<std::mutex> render_lock(_mtx);

  CHECK(_perGpuData.size());

  // uses the first gpu as the default.
  // TODO(croot): expose a way to specify which gpu to use?
  // auto itr = _perGpuData.begin();
  // _activeRenderer->activateGpus(_perGpuData, itr->first);
  _activeRenderer->activateGpus(_perGpuData);

  _activeRenderer->setJSONDocument(jsonDocumentPtr, false);
}
#endif  // HAVE_CUDA

void QueryRenderManager::setWidthHeight(int width, int height) {
  // RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
  //                   "setWidthHeight: There is no active user/widget id. Must set an active user/widget id before "
  //                   "setting width/height.");

  // std::lock_guard<std::mutex> render_lock(_mtx);
  // glfwMakeContextCurrent(_windowPtr);
  // _activeRenderer->setWidthHeight(width, height, (_debugMode ? _windowPtr : nullptr));
  // glfwMakeContextCurrent(nullptr);
}

void QueryRenderManager::render() {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "render(): There is no active user/widget id. Must set a user/widget id active before rendering.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  _activeRenderer->render();

  // if (_debugMode) {
  //   MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
  //   MAPD_CHECK_GL_ERROR(glDrawBuffer(GL_BACK));

  //   // TODO(croot): need an API to set the framebuffer's read buffer
  //   _activeRenderer->getFramebuffer()->bindToRenderer(BindType::READ);
  //   MAPD_CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));

  //   int framebufferWidth, framebufferHeight;
  //   // glfwGetFramebufferSize(_windowPtr.get(), &framebufferWidth, &framebufferHeight);
  //   MAPD_CHECK_GL_ERROR(glfwGetFramebufferSize(_windowPtr, &framebufferWidth, &framebufferHeight));

  //   MAPD_CHECK_GL_ERROR(glBlitFramebuffer(0,
  //                                         0,
  //                                         framebufferWidth,
  //                                         framebufferHeight,
  //                                         0,
  //                                         0,
  //                                         framebufferWidth,
  //                                         framebufferHeight,
  //                                         GL_COLOR_BUFFER_BIT,
  //                                         GL_NEAREST));
  // }
}

PngData QueryRenderManager::renderToPng(int compressionLevel) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "There is no active user/widget id. Must set a user/widget id active before rendering.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  return _activeRenderer->renderToPng(compressionLevel);
}

int64_t QueryRenderManager::getIdAt(int x, int y) {
  RUNTIME_EX_ASSERT(_activeRenderer != nullptr,
                    "getIdAt(): There is no active user/widget id. Must set an active user/widget id before "
                    "requesting pixel data.");

  std::lock_guard<std::mutex> render_lock(_mtx);
  int64_t id = _activeRenderer->getIdAt(x, y);

  // ids go from 0 to numitems-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
  return id - 1;
}

}  // namespace QueryRenderer
