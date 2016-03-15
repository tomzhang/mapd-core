#include "QueryRenderManager.h"
#include "QueryRenderer.h"
#include <Rendering/WindowManager.h>
#include <Rendering/Settings/WindowSettings.h>
#include <Rendering/Settings/RendererSettings.h>
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <unordered_map>
#include <boost/lambda/lambda.hpp>

namespace QueryRenderer {

using ::Rendering::Settings::WindowSettings;
using ::Rendering::Settings::RendererSettings;

using ::Rendering::Settings::IntConstant;
using ::Rendering::Settings::IntSetting;
using ::Rendering::Settings::StrSetting;

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;

static std::chrono::milliseconds getCurrentTimeMS() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
}

const UserWidgetPair QueryRenderManager::_emptyUserWidget = std::make_pair(-1, -1);

const std::chrono::milliseconds QueryRenderManager::maxWidgetIdleTime =
    std::chrono::milliseconds(300000);  // 5 minutes, in ms

QueryRenderManager::SessionData::SessionData(int userId, int widgetId, QueryRenderer* newRenderer)
    : userId(userId), widgetId(widgetId), renderer(newRenderer) {
  lastRenderTime = getCurrentTimeMS();
}

QueryRenderManager::ChangeLastRenderTime::ChangeLastRenderTime() {
  new_time = getCurrentTimeMS();
}

void QueryRenderManager::ChangeLastRenderTime::operator()(SessionData& sd) {
  sd.lastRenderTime = new_time;
}

QueryRenderManager::QueryRenderManager(int numGpus, int startGpu, size_t queryResultBufferSize, size_t renderCacheLimit)
    : _rendererMap(), _activeItr(_rendererMap.end()), _perGpuData(), _renderCacheLimit(renderCacheLimit) {
  // NOTE: this constructor needs to be used on the main thread as that is a requirement
  // for a window manager instance.
  ::Rendering::WindowManager windowMgr;
  _initialize(windowMgr, numGpus, startGpu, queryResultBufferSize);
}

QueryRenderManager::QueryRenderManager(Rendering::WindowManager& windowMgr,
                                       int numGpus,
                                       int startGpu,
                                       size_t queryResultBufferSize,
                                       size_t renderCacheLimit)
    : _rendererMap(), _activeItr(_rendererMap.end()), _perGpuData(), _renderCacheLimit(renderCacheLimit) {
  _initialize(windowMgr, numGpus, startGpu, queryResultBufferSize);
}

QueryRenderManager::~QueryRenderManager() {
}

void QueryRenderManager::_initialize(Rendering::WindowManager& windowMgr,
                                     int numGpus,
                                     int startGpu,
                                     size_t queryResultBufferSize) {
  int maxNumGpus = windowMgr.getNumGpus();

  if (numGpus <= 0) {
    numGpus = maxNumGpus - startGpu;
    startGpu = 0;  // if using all available gpus, we must start on gpu 0
  } else {
    RUNTIME_EX_ASSERT(startGpu < maxNumGpus,
                      "Invalid start GPU: " + std::to_string(startGpu) + ". There is only " +
                          std::to_string(maxNumGpus) + " GPU" + (maxNumGpus > 1 ? "s" : "") + " available.");
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

  GLRenderer* renderer = nullptr;
  GLResourceManagerShPtr rsrcMgrPtr;

  PerGpuDataShPtr gpuDataPtr;
  int endGpu = startGpu + numGpus;
  for (int i = startGpu; i < endGpu; ++i) {
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

    // make sure to clear the renderer from the current thread
    gpuDataPtr->makeInactive();
  }

  LOG(INFO) << "QueryRenderManager initialized for rendering. start GPU: " << startGpu << ", num GPUs: " << endGpu
            << ", Render cache limit: " << _renderCacheLimit;
}

void QueryRenderManager::_resetQueryResultBuffers() {
  for (auto& itr : _perGpuData) {
    itr.second->queryResultBufferPtr->reset();
  }
}

void QueryRenderManager::setActiveUserWidget(int userId, int widgetId) {
  // purge any idle users

  {
    std::lock_guard<std::mutex> user_lock(_usersMtx);

    // if (userWidget != _activeUserWidget) {
    if (_activeItr == _rendererMap.end() || (userId != _activeItr->userId || widgetId != _activeItr->widgetId)) {
      auto itr = _rendererMap.find(std::make_tuple(userId, widgetId));

      RUNTIME_EX_ASSERT(
          itr != _rendererMap.end(),
          "User id: " + std::to_string(userId) + ", Widget Id: " + std::to_string(widgetId) + " does not exist.");

      _activeItr = itr;
    }

    _updateActiveLastRenderTime();
  }

  _purgeUnusedWidgets();
}

void QueryRenderManager::setActiveUserWidget(const UserWidgetPair& userWidgetPair) {
  setActiveUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

bool QueryRenderManager::hasUser(int userId) const {
  std::lock_guard<std::mutex> user_lock(_usersMtx);

  auto& userIdMap = _rendererMap.get<UserId>();

  return (userIdMap.find(userId) != userIdMap.end());
}

bool QueryRenderManager::hasUserWidget(int userId, int widgetId) const {
  std::lock_guard<std::mutex> user_lock(_usersMtx);
  return (_rendererMap.find(std::make_tuple(userId, widgetId)) != _rendererMap.end());
}

bool QueryRenderManager::hasUserWidget(const UserWidgetPair& userWidgetPair) const {
  return hasUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

void QueryRenderManager::addUserWidget(int userId, int widgetId, bool doHitTest, bool doDepthTest) {
  std::lock_guard<std::mutex> user_lock(_usersMtx);

  RUNTIME_EX_ASSERT(_rendererMap.find(std::make_tuple(userId, widgetId)) == _rendererMap.end(),
                    "Cannot add user widget. User id: " + std::to_string(userId) + " with widget id: " +
                        std::to_string(widgetId) + " already exists.");

  // Check if the current num of connections is maxed out, NOTE: can only add 1 at a time
  // here
  if (_rendererMap.size() == _renderCacheLimit) {
    auto& lastRenderTimeList = _rendererMap.get<LastRenderTime>();

    auto itr = lastRenderTimeList.begin();

    if (_activeItr != _rendererMap.end()) {
      auto activeItr = _rendererMap.project<LastRenderTime>(_activeItr);

      if (itr == activeItr) {
        // We won't remove the active itr if it is the most idle one
        // unless the renderCacheLimit == 1, in which case the
        // active itr is forcibly removed.
        // TODO(croot): Should we always remove the active itr?
        if (_renderCacheLimit == 1) {
          _clearActiveUserWidget();
        } else {
          itr++;
        }
      }
    }

    LOG(INFO) << "QueryRenderManager render cache limit reached. Removing longest-idle connection (" << itr->userId
              << ", " << itr->widgetId << ")";
    lastRenderTimeList.erase(itr);
  }

  _rendererMap.emplace(userId, widgetId, new QueryRenderer(doHitTest, doDepthTest));
}

void QueryRenderManager::addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest, bool doDepthTest) {
  addUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair), doHitTest, doDepthTest);
}

void QueryRenderManager::removeUserWidget(int userId, int widgetId) {
  std::lock_guard<std::mutex> user_lock(_usersMtx);
  auto itr = _rendererMap.find(std::make_tuple(userId, widgetId));

  RUNTIME_EX_ASSERT(itr != _rendererMap.end(),
                    "User id: " + std::to_string(userId) + "Widget id: " + std::to_string(widgetId) +
                        " does not exist. Cannot remove their caches.");

  _rendererMap.erase(itr);

  if (itr == _activeItr) {
    _clearActiveUserWidget();
  }
}

void QueryRenderManager::removeUserWidget(const UserWidgetPair& userWidgetPair) {
  removeUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

// Removes all widgets/sessions for a particular user id.
void QueryRenderManager::removeUser(int userId) {
  std::lock_guard<std::mutex> user_lock(_usersMtx);

  auto& userIdMap = _rendererMap.get<UserId>();

  auto startEndItr = userIdMap.equal_range(userId);
  RUNTIME_EX_ASSERT(startEndItr.first != userIdMap.end(),
                    "User id " + std::to_string(userId) + " does not exist. Cannot remove its caches.");

  if (userId == _activeItr->userId) {
    _clearActiveUserWidget();
  }

  userIdMap.erase(startEndItr.first, startEndItr.second);
}

void QueryRenderManager::_clearActiveUserWidget() {
  _activeItr = _rendererMap.end();
}

void QueryRenderManager::_purgeUnusedWidgets() {
  std::lock_guard<std::mutex> user_lock(_usersMtx);

  std::chrono::milliseconds cutoffTime = getCurrentTimeMS() - maxWidgetIdleTime;

  // the currently active itr should not be a purgable
  CHECK(_activeItr == _rendererMap.end() || _activeItr->lastRenderTime >= cutoffTime);

  auto& lastRenderTimeList = _rendererMap.get<LastRenderTime>();

  int cnt = 0;
  auto itr = lastRenderTimeList.begin();
  while (itr != lastRenderTimeList.end() && itr->lastRenderTime < cutoffTime) {
    cnt++;
    itr++;
  }

  LOG_IF(INFO, cnt > 0) << "QueryRenderManager - purging " << cnt << " idle connections.";
  lastRenderTimeList.erase(lastRenderTimeList.begin(), itr);
}

void QueryRenderManager::_updateActiveLastRenderTime() {
  if (_activeItr == _rendererMap.end()) {
    return;
  }

  _rendererMap.modify(_activeItr, ChangeLastRenderTime());

  auto& lastRenderTimeList = _rendererMap.get<LastRenderTime>();
  auto lastRenderTimeItr = _rendererMap.project<LastRenderTime>(_activeItr);

  lastRenderTimeList.relocate(lastRenderTimeList.end(), lastRenderTimeItr);
}

#ifdef HAVE_CUDA
CudaHandle QueryRenderManager::getCudaHandle(const GpuId& gpuId) {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr != _perGpuData.end(), "Cannot get cuda handle for gpu " + std::to_string(gpuId) + ".");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  itr->second->makeActiveOnCurrentThread();
  CudaHandle rtn = itr->second->queryResultBufferPtr->getCudaHandlePreQuery();
  itr->second->rendererPtr->makeInactive();

  return rtn;
}

void QueryRenderManager::setCudaHandleUsedBytes(GpuId gpuId, size_t numUsedBytes) {
  auto itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr != _perGpuData.end(), "Cannot set cuda handle results for gpu " + std::to_string(gpuId) + ".");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  itr->second->makeActiveOnCurrentThread();
  itr->second->queryResultBufferPtr->updatePostQuery(numUsedBytes);
  itr->second->rendererPtr->makeInactive();
}

void QueryRenderManager::configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                                         QueryDataLayoutShPtr dataLayoutPtr,
                                         const Executor* executor) {
  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "ConfigureRender: There is no active user/widget id. Must set a user/widget id active before "
                    "configuring the render.");

  std::lock_guard<std::mutex> render_lock(_renderMtx);

  // need to update the data layout of the query result buffer before building up
  // from the json obj
  if (dataLayoutPtr) {
    // CHECK(executor != nullptr);

    _activeItr->renderer->updateResultsPostQuery(dataLayoutPtr, executor, _perGpuData);
  } else {
    CHECK(_perGpuData.size());

    _activeItr->renderer->activateGpus(_perGpuData);
  }

  _activeItr->renderer->setJSONDocument(jsonDocumentPtr, false);
}
#else
void QueryRenderManager::configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr) {
  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "ConfigureRender: There is no active user/widget id. Must set a user/widget id active before "
                    "configuring the render.");

  std::lock_guard<std::mutex> render_lock(_renderMtx);

  CHECK(_perGpuData.size());

  _activeItr->renderer->activateGpus(_perGpuData);

  _activeItr->renderer->setJSONDocument(jsonDocumentPtr, false);
}
#endif  // HAVE_CUDA

void QueryRenderManager::setWidthHeight(int width, int height) {
  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "setWidthHeight: There is no active user/widget id. Must set an active user/widget id before "
                    "setting width/height.");

  std::lock_guard<std::mutex> render_lock(_renderMtx);
  _activeItr->renderer->setWidthHeight(width, height);
}

std::vector<GpuId> QueryRenderManager::getAllGpuIds() const {
  std::vector<GpuId> rtn;
  for (auto itr : _perGpuData) {
    rtn.push_back(itr.first);
  }

  return rtn;
}

void QueryRenderManager::render() {
  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "render(): There is no active user/widget id. Must set a user/widget id active before rendering.");

  std::lock_guard<std::mutex> render_lock(_renderMtx);

  _activeItr->renderer->render();
  _updateActiveLastRenderTime();

  _resetQueryResultBuffers();
}

PngData QueryRenderManager::renderToPng(int compressionLevel) {
  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "There is no active user/widget id. Must set a user/widget id active before rendering.");

  std::lock_guard<std::mutex> render_lock(_renderMtx);

  PngData rtn = _activeItr->renderer->renderToPng(compressionLevel);

  _updateActiveLastRenderTime();

  _resetQueryResultBuffers();

  return rtn;
}

int64_t QueryRenderManager::getIdAt(size_t x, size_t y) {
  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "getIdAt(): There is no active user/widget id. Must set an active user/widget id before "
                    "requesting pixel data.");

  std::lock_guard<std::mutex> render_lock(_renderMtx);
  int64_t id = _activeItr->renderer->getIdAt(x, y);
  _updateActiveLastRenderTime();

  // ids go from 0 to numitems-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
  return id - 1;
}

}  // namespace QueryRenderer
