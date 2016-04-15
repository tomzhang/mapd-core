#include "QueryRenderManager.h"
#include "QueryRenderer.h"
#include "Utilities.h"
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
using ::Rendering::GL::GLRendererShPtr;
using ::Rendering::GL::GLResourceManagerShPtr;

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

QueryRenderManager::ActiveRendererGuard::ActiveRendererGuard(QueryRenderManager::PerGpuData* currGpuData,
                                                             QueryRenderManager* qrm)
    : currGpuData(currGpuData), qrm(qrm) {
  if (currGpuData) {
    currGpuData->makeActiveOnCurrentThread();
  }
}

QueryRenderManager::ActiveRendererGuard::~ActiveRendererGuard() {
  ::Rendering::GL::GLRenderer* renderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
  if (renderer) {
    renderer->makeInactive();
  }

  if (qrm) {
    qrm->_resetQueryResultBuffers();
  }
}

QueryRenderManager::QueryRenderManager(CudaMgr_Namespace::CudaMgr* cudaMgr,
                                       int numGpus,
                                       int startGpu,
                                       size_t queryResultBufferSize,
                                       size_t renderCacheLimit)
    : _rendererMap(),
      _activeItr(_rendererMap.end()),
      _perGpuData(new PerGpuDataMap()),
      _compositorPtr(nullptr),
      _renderCacheLimit(renderCacheLimit) {
  // NOTE: this constructor needs to be used on the main thread as that is a requirement
  // for a window manager instance.
  ::Rendering::WindowManager windowMgr;
  _initialize(windowMgr, cudaMgr, numGpus, startGpu, queryResultBufferSize);
}

QueryRenderManager::QueryRenderManager(Rendering::WindowManager& windowMgr,
                                       CudaMgr_Namespace::CudaMgr* cudaMgr,
                                       int numGpus,
                                       int startGpu,
                                       size_t queryResultBufferSize,
                                       size_t renderCacheLimit)
    : _rendererMap(),
      _activeItr(_rendererMap.end()),
      _perGpuData(),
      _compositorPtr(nullptr),
      _renderCacheLimit(renderCacheLimit) {
  _initialize(windowMgr, cudaMgr, numGpus, startGpu, queryResultBufferSize);
}

QueryRenderManager::~QueryRenderManager() {
}

void QueryRenderManager::_initialize(Rendering::WindowManager& windowMgr,
                                     CudaMgr_Namespace::CudaMgr* cudaMgr,
                                     int numGpus,
                                     int startGpu,
                                     size_t queryResultBufferSize) {
  int maxNumGpus = windowMgr.getNumGpus();

  if (numGpus <= 0) {
    numGpus = maxNumGpus - startGpu;
    startGpu = 0;  // if using all available gpus, we must start on gpu 0
  } else {
    RUNTIME_EX_ASSERT(startGpu < maxNumGpus,
                      "QueryRenderManager initialization - Invalid start GPU: " + std::to_string(startGpu) +
                          ". There is only " + std::to_string(maxNumGpus) + " GPU" + (maxNumGpus > 1 ? "s" : "") +
                          " available.");
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

  GLRendererShPtr renderer;
  GLResourceManagerShPtr rsrcMgrPtr;

  PerGpuDataShPtr gpuDataPtr;
  size_t endDevice = startGpu + numGpus;
  size_t startDevice = startGpu;

  ActiveRendererGuard renderGuard;

  for (size_t i = startDevice; i < endDevice; ++i) {
    if (cudaMgr) {
      // need to set a cuda context before creating gl/cuda interop buffers
      cudaMgr->setContext(i);
    }

    windowSettings.setStrSetting(StrSetting::NAME, windowName + std::to_string(i));
    windowSettings.setIntSetting(IntSetting::GPU_ID, i);

    auto itr = _perGpuData->emplace(new PerGpuData(i));

    gpuDataPtr = *itr.first;
    gpuDataPtr->windowPtr = windowMgr.createWindow(windowSettings);
    gpuDataPtr->rendererPtr = windowMgr.createRendererForWindow(rendererSettings, gpuDataPtr->windowPtr);

    gpuDataPtr->makeActiveOnCurrentThread();

    if (i == startDevice && numGpus > 1) {
      _compositorPtr.reset(new QueryRenderCompositor(this,
                                                     gpuDataPtr->rendererPtr,
                                                     defaultWidth,
                                                     defaultHeight,
                                                     // TODO(croot): should we support num samples
                                                     1,  //_ctx->getNumSamples(),
                                                     true,
                                                     // TODO(croot): do depth testing
                                                     false));
    }

    // renderer = dynamic_cast<GLRenderer*>(gpuDataPtr->rendererPtr.get());
    renderer = std::static_pointer_cast<GLRenderer>(gpuDataPtr->rendererPtr);
    CHECK(renderer != nullptr);

    gpuDataPtr->queryResultBufferPtr.reset(new QueryResultVertexBuffer(renderer.get(), queryResultBufferSize));

    if (_compositorPtr) {
      gpuDataPtr->compositorPtr = _compositorPtr;
      gpuDataPtr->framebufferPtr.reset(new QueryFramebuffer(_compositorPtr.get(), renderer.get()));
    } else {
      // TODO(croot): do depth testing
      gpuDataPtr->framebufferPtr.reset(new QueryFramebuffer(renderer.get(), defaultWidth, defaultHeight, true, false));
    }

    gpuDataPtr->pboPoolPtr.reset(new QueryIdMapPboPool(renderer));

    // make sure to clear the renderer from the current thread
    gpuDataPtr->makeInactive();
  }

  LOG(INFO) << "QueryRenderManager initialized for rendering. start GPU: " << startDevice << ", num GPUs: " << endDevice
            << ", Render cache limit: " << _renderCacheLimit;
}

void QueryRenderManager::_resetQueryResultBuffers() noexcept {
  for (auto& itr : *_perGpuData) {
    itr->queryResultBufferPtr->reset();
  }
}

void QueryRenderManager::setActiveUserWidget(int userId, int widgetId) {
  // purge any idle users

  {
    std::lock_guard<std::mutex> render_lock(_renderMtx);

    if (_activeItr == _rendererMap.end() || (userId != _activeItr->userId || widgetId != _activeItr->widgetId)) {
      auto itr = _rendererMap.find(std::make_tuple(userId, widgetId));

      RUNTIME_EX_ASSERT(
          itr != _rendererMap.end(),
          "User id: " + std::to_string(userId) + ", Widget Id: " + std::to_string(widgetId) + " does not exist.");

      _activeItr = itr;
      LOG(INFO) << "Active render session [userId: " << _activeItr->userId << ", widgetId: " << _activeItr->widgetId
                << "]";
    }

    _updateActiveLastRenderTime();
  }

  _purgeUnusedWidgets();
}

void QueryRenderManager::setActiveUserWidget(const UserWidgetPair& userWidgetPair) {
  setActiveUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

bool QueryRenderManager::hasUser(int userId) const {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  auto& userIdMap = _rendererMap.get<UserId>();

  return (userIdMap.find(userId) != userIdMap.end());
}

bool QueryRenderManager::hasUserWidget(int userId, int widgetId) const {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  return (_rendererMap.find(std::make_tuple(userId, widgetId)) != _rendererMap.end());
}

bool QueryRenderManager::hasUserWidget(const UserWidgetPair& userWidgetPair) const {
  return hasUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

void QueryRenderManager::addUserWidget(int userId, int widgetId, bool doHitTest, bool doDepthTest) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_rendererMap.find(std::make_tuple(userId, widgetId)) == _rendererMap.end(),
                    "Cannot add user widget. User id: " + std::to_string(userId) + " with widget id: " +
                        std::to_string(widgetId) + " already exists.");

  ActiveRendererGuard activeRendererGuard;

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

  _rendererMap.emplace(userId, widgetId, new QueryRenderer(userId, widgetId, _perGpuData, doHitTest, doDepthTest));
}

void QueryRenderManager::addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest, bool doDepthTest) {
  addUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair), doHitTest, doDepthTest);
}

void QueryRenderManager::removeUserWidget(int userId, int widgetId) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  auto itr = _rendererMap.find(std::make_tuple(userId, widgetId));

  RUNTIME_EX_ASSERT(itr != _rendererMap.end(),
                    "User id: " + std::to_string(userId) + "Widget id: " + std::to_string(widgetId) +
                        " does not exist. Cannot remove their caches.");

  ActiveRendererGuard activeRendererGuard;

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
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  auto& userIdMap = _rendererMap.get<UserId>();

  auto startEndItr = userIdMap.equal_range(userId);
  RUNTIME_EX_ASSERT(startEndItr.first != userIdMap.end(),
                    "User id " + std::to_string(userId) + " does not exist. Cannot remove its caches.");

  ActiveRendererGuard activeRendererGuard;

  if (userId == _activeItr->userId) {
    _clearActiveUserWidget();
  }

  userIdMap.erase(startEndItr.first, startEndItr.second);
}

void QueryRenderManager::_clearActiveUserWidget() {
  _activeItr = _rendererMap.end();
  LOG(INFO) << "Active render session [userId: -1, widgetId: -1] (unset)";
}

void QueryRenderManager::_purgeUnusedWidgets() {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

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

  ActiveRendererGuard activeRendererGuard;
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

CudaHandle QueryRenderManager::getCudaHandle(size_t gpuIdx) {
#ifdef HAVE_CUDA
  PerGpuDataMap_in_order& inOrder = _perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "Cannot get cuda handle for gpu index " + std::to_string(gpuIdx) + ". There are only " +
                        std::to_string(inOrder.size()) + " gpus available.");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  ActiveRendererGuard activeRendererGuard(inOrder[gpuIdx].get());
  CudaHandle rtn = inOrder[gpuIdx]->queryResultBufferPtr->getCudaHandlePreQuery();

  return rtn;
#else
  CHECK(false) << "Cuda is not activated. Cannot get cuda handle.";
#endif  // HAVE_CUDA
}

void QueryRenderManager::setCudaHandleUsedBytes(size_t gpuIdx, size_t numUsedBytes) {
#ifdef HAVE_CUDA
  PerGpuDataMap_in_order& inOrder = _perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "Cannot set cuda handle results for gpu index " + std::to_string(gpuIdx) + ". There are only " +
                        std::to_string(inOrder.size()) + " gpus available.");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  ActiveRendererGuard activeRendererGuard(inOrder[gpuIdx].get());
  inOrder[gpuIdx]->queryResultBufferPtr->updatePostQuery(numUsedBytes);
#else
  CHECK(false) << "Cuda is not activated. Cannot set cuda handle bytes.";
#endif  // HAVE_CUDA
}

void QueryRenderManager::configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                                         QueryDataLayoutShPtr dataLayoutPtr,
                                         const Executor* executor) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "ConfigureRender: There is no active user/widget id. Must set a user/widget id active before "
                    "configuring the render.");

  ActiveRendererGuard activeRendererGuard(nullptr, this);

  // need to update the data layout of the query result buffer before building up
  // from the json obj
  if (dataLayoutPtr) {
    _activeItr->renderer->updateResultsPostQuery(dataLayoutPtr, executor);
  } else {
    CHECK(_perGpuData->size());

    _activeItr->renderer->activateGpus();
  }

  _activeItr->renderer->setJSONDocument(jsonDocumentPtr, false);
}

void QueryRenderManager::setWidthHeight(int width, int height) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "setWidthHeight: There is no active user/widget id. Must set an active user/widget id before "
                    "setting width/height.");

  ActiveRendererGuard activeRendererGuard;

  _activeItr->renderer->setWidthHeight(width, height);
}

size_t QueryRenderManager::getNumGpus() const {
  return _perGpuData->size();
}

std::vector<GpuId> QueryRenderManager::getAllGpuIds() const {
  std::vector<GpuId> rtn;
  for (auto itr : *_perGpuData) {
    rtn.push_back(itr->gpuId);
  }

  return rtn;
}

void QueryRenderManager::render() {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "render(): There is no active user/widget id. Must set a user/widget id active before rendering.");

  ActiveRendererGuard activeRendererGuard;

  _activeItr->renderer->render();
  _updateActiveLastRenderTime();
}

PngData QueryRenderManager::renderToPng(int compressionLevel) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "There is no active user/widget id. Must set a user/widget id active before rendering.");

  ActiveRendererGuard activeRendererGuard;

  PngData rtn = _activeItr->renderer->renderToPng(compressionLevel);

  _updateActiveLastRenderTime();

  return rtn;
}

int64_t QueryRenderManager::getIdAt(size_t x, size_t y, size_t pixelRadius) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "getIdAt(): There is no active user/widget id. Must set an active user/widget id before "
                    "requesting pixel data.");

  ActiveRendererGuard activeRendererGuard;
  int64_t id = _activeItr->renderer->getIdAt(x, y, pixelRadius);
  _updateActiveLastRenderTime();

  // ids go from 0 to numitems-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
  return id - 1;
}

}  // namespace QueryRenderer
