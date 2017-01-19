#include "QueryRenderManager.h"
#include "QueryRenderer.h"
#include "Utils/Utils.h"
#include "Rendering/QueryRenderSMAAPass.h"
#include <Rendering/WindowManager.h>
#include <Rendering/Settings/WindowSettings.h>
#include <Rendering/Settings/RendererSettings.h>
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <QueryEngine/Execute.h>
#include <Shared/measure.h>

#include <unordered_map>
#include <boost/lambda/lambda.hpp>

#include "rapidjson/error/en.h"

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

QueryRenderManager::ActiveRendererGuard::ActiveRendererGuard(RootPerGpuData* currGpuData, QueryRenderManager* qrm)
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

  // if (qrm) {
  //   qrm->_resetQueryResultBuffers();
  // }
}

QueryRenderManager::QueryRenderManager(CudaMgr_Namespace::CudaMgr* cudaMgr,
                                       int numGpus,
                                       int startGpu,
                                       size_t queryResultBufferSize,
                                       size_t renderCacheLimit,
                                       size_t numSamples)
    : _rendererMap(),
      _activeItr(_rendererMap.end()),
      _gpuCache(new RootCache(numSamples)),
      _compositorPtr(nullptr),
      _renderCacheLimit(renderCacheLimit) {
  // NOTE: this constructor needs to be used on the main thread as that is a requirement
  // for a window manager instance.
  ::Rendering::WindowManager windowMgr;
  _initialize(windowMgr, cudaMgr, numGpus, startGpu, queryResultBufferSize, numSamples);
}

QueryRenderManager::QueryRenderManager(Rendering::WindowManager& windowMgr,
                                       CudaMgr_Namespace::CudaMgr* cudaMgr,
                                       int numGpus,
                                       int startGpu,
                                       size_t queryResultBufferSize,
                                       size_t renderCacheLimit,
                                       size_t numSamples)
    : _rendererMap(),
      _activeItr(_rendererMap.end()),
      _gpuCache(new RootCache(numSamples)),
      _compositorPtr(nullptr),
      _renderCacheLimit(renderCacheLimit) {
  _initialize(windowMgr, cudaMgr, numGpus, startGpu, queryResultBufferSize, numSamples);
}

QueryRenderManager::~QueryRenderManager() {}

void QueryRenderManager::_initialize(Rendering::WindowManager& windowMgr,
                                     CudaMgr_Namespace::CudaMgr* cudaMgr,
                                     int numGpus,
                                     int startGpu,
                                     size_t queryResultBufferSize,
                                     size_t numSamples) {
#ifdef HAVE_CUDA
  CHECK(cudaMgr);
#endif

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

#ifdef MAPDGL_EGL
  if (numSamples > 1 && numGpus > 1) {
    LOG(WARNING) << "QueryRenderManager: initializing the render manager with " << numGpus
                 << " gpus and the compositing will be performed using EGLImage objects. EGLImage objects do not "
                    "support multisampling, so forcing the number of samples from "
                 << numSamples << " to 1.";
    numSamples = 1;
  }
#endif

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
  RootPerGpuDataShPtr gpuDataPtr;
  size_t endDevice = startGpu + numGpus;
  size_t startDevice = startGpu;

  ActiveRendererGuard renderGuard;
  std::unordered_map<std::string, bool> extensionsToUse = {{"GL_NV_vertex_attrib_integer_64bit", false}};

  // first build out the rendering contexts on each GPU and build up the supported extensions
  for (size_t i = startDevice; i < endDevice; ++i) {
    windowSettings.setStrSetting(StrSetting::NAME, windowName + std::to_string(i));
    windowSettings.setIntSetting(IntSetting::GPU_ID, i);

    auto itr = _gpuCache->perGpuData->emplace(new RootPerGpuData(i));

    gpuDataPtr = *itr.first;
    gpuDataPtr->windowPtr = windowMgr.createWindow(windowSettings);
    gpuDataPtr->rendererPtr = windowMgr.createRendererForWindow(rendererSettings, gpuDataPtr->windowPtr);

    renderer = std::dynamic_pointer_cast<GLRenderer>(gpuDataPtr->rendererPtr);
    CHECK(renderer != nullptr);

    gpuDataPtr->makeActiveOnCurrentThread();

    // now check extensions
    if (i == startDevice) {
      for (auto& item : extensionsToUse) {
        if (renderer->supportsExtension(item.first)) {
          _gpuCache->supportedExtensions.insert(item.first);
          item.second = true;
        } else {
          item.second = false;
        }
      }
    } else {
      for (auto& item : extensionsToUse) {
        if (item.second && !renderer->supportsExtension(item.first)) {
          _gpuCache->supportedExtensions.erase(item.first);
          item.second = false;
        }
      }
    }

    // make sure to clear the renderer from the current thread
    gpuDataPtr->makeInactive();
  }

  bool supportsInt64 = _gpuCache->supportsInt64();

  // now build the global gpu resources. This is done after all contexts have been built
  // so we know what extensions are supported on all.
  for (size_t i = startDevice; i < endDevice; ++i) {
    if (cudaMgr) {
      // need to set a cuda context before creating gl/cuda interop buffers
      cudaMgr->setContext(i - startDevice);
    }

    auto itr = _gpuCache->perGpuData->find(i);
    CHECK(itr != _gpuCache->perGpuData->end());
    gpuDataPtr = *itr;

    renderer = std::dynamic_pointer_cast<GLRenderer>(gpuDataPtr->rendererPtr);
    CHECK(renderer != nullptr);

    gpuDataPtr->makeActiveOnCurrentThread();

    if (i == startDevice) {
      if (numGpus > 1) {
        _compositorPtr.reset(new QueryRenderCompositor(this,
                                                       gpuDataPtr->rendererPtr,
                                                       defaultWidth,
                                                       defaultHeight,
                                                       numSamples,
                                                       true,
                                                       // TODO(croot): do depth testing
                                                       false,
                                                       supportsInt64));
      }
    }

    gpuDataPtr->queryResultBufferPtr.reset(new QueryResultVertexBuffer(renderer.get(), queryResultBufferSize));

    if (_compositorPtr) {
      gpuDataPtr->compositorPtr = _compositorPtr;
      gpuDataPtr->msFramebufferPtr.reset(new QueryFramebuffer(_compositorPtr.get(), renderer.get()));
    } else {
      // TODO(croot): do depth testing
      gpuDataPtr->msFramebufferPtr.reset(
          new QueryFramebuffer(renderer.get(), defaultWidth, defaultHeight, true, false, numSamples));
    }

    gpuDataPtr->pboPoolUIntPtr.reset(new QueryIdMapPboPool<unsigned int>(renderer));
    gpuDataPtr->pboPoolIntPtr.reset(new QueryIdMapPboPool<int>(renderer));

    // create a framebuffer for anti-aliasing.
    // If multi-sampling is enabled, this is used to blit the multi-sampled framebuffers into
    // Otherwise it's used to store the output of an anti-aliasing post-processing pass
    gpuDataPtr->aaFramebufferPtr.reset(
        new QueryFramebuffer(renderer.get(), defaultWidth, defaultHeight, true, false, 1));

    gpuDataPtr->smaaPassPtr.reset(new QueryRenderSMAAPass(renderer,
                                                          defaultWidth,
                                                          defaultHeight,
                                                          numSamples,
                                                          QueryRenderSMAAPass::SMAA_QUALITY_PRESET::HIGH,
                                                          QueryRenderSMAAPass::SMAA_EDGE_DETECTION_TYPE::LUMA));

    // make sure to clear the renderer from the current thread
    gpuDataPtr->makeInactive();
  }

  LOG(INFO) << "QueryRenderManager initialized for rendering. start GPU: " << startDevice << ", num GPUs: " << numGpus
            << ", Render cache limit: " << _renderCacheLimit;
}

void QueryRenderManager::_resetQueryResultBuffers() noexcept {
  for (auto& itr : *(_gpuCache->perGpuData)) {
    itr->queryResultBufferPtr->reset();
  }

  // for (auto& cacheItr : _gpuCache->polyCacheMap) {
  //   cacheItr.second.reset();
  // }
}

void QueryRenderManager::_setActiveUserWidgetInternal(int userId, int widgetId) {
  // purge any idle users
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

  _purgeUnusedWidgets();
}

bool QueryRenderManager::_hasUserInternal(int userId) const {
  auto& userIdMap = _rendererMap.get<UserId>();

  return (userIdMap.find(userId) != userIdMap.end());
}

bool QueryRenderManager::_hasUserWidgetInternal(int userId, int widgetId) const {
  return (_rendererMap.find(std::make_tuple(userId, widgetId)) != _rendererMap.end());
}

void QueryRenderManager::_addUserWidgetInternal(int userId, int widgetId, bool doHitTest, bool doDepthTest) {
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

  _rendererMap.emplace(userId, widgetId, new QueryRenderer(userId, widgetId, _gpuCache, doHitTest, doDepthTest));
}

void QueryRenderManager::_removeUserWidgetInternal(int userId, int widgetId) {
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

void QueryRenderManager::_removeUserInternal(int userId) {
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

void QueryRenderManager::setActiveUserWidget(int userId, int widgetId) {
  // DEPRECATED

  // purge any idle users
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  return _setActiveUserWidgetInternal(userId, widgetId);
}

void QueryRenderManager::setActiveUserWidget(const UserWidgetPair& userWidgetPair) {
  // DEPRECATED
  setActiveUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

bool QueryRenderManager::hasUser(int userId) const {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  return _hasUserInternal(userId);
}

bool QueryRenderManager::hasUserWidget(int userId, int widgetId) const {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  return _hasUserWidgetInternal(userId, widgetId);
}

bool QueryRenderManager::hasUserWidget(const UserWidgetPair& userWidgetPair) const {
  return hasUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

void QueryRenderManager::addUserWidget(int userId, int widgetId, bool doHitTest, bool doDepthTest) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  return _addUserWidgetInternal(userId, widgetId, doHitTest, doDepthTest);
}

void QueryRenderManager::addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest, bool doDepthTest) {
  addUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair), doHitTest, doDepthTest);
}

void QueryRenderManager::removeUserWidget(int userId, int widgetId) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  return _removeUserWidgetInternal(userId, widgetId);
}

void QueryRenderManager::removeUserWidget(const UserWidgetPair& userWidgetPair) {
  removeUserWidget(std::get<0>(userWidgetPair), std::get<1>(userWidgetPair));
}

// Removes all widgets/sessions for a particular user id.
void QueryRenderManager::removeUser(int userId) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  return _removeUserInternal(userId);
}

void QueryRenderManager::_clearActiveUserWidget() {
  _activeItr = _rendererMap.end();
  LOG(INFO) << "Active render session [userId: -1, widgetId: -1] (unset)";
}

void QueryRenderManager::_purgeUnusedWidgets() {
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
  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "Cannot get cuda handle for gpu index " + std::to_string(gpuIdx) + ". There are only " +
                        std::to_string(inOrder.size()) + " gpus available.");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_bufferMtx);

  ActiveRendererGuard activeRendererGuard(inOrder[gpuIdx].get());

  inOrder[gpuIdx]->queryResultBufferPtr->reset();
  CudaHandle rtn = inOrder[gpuIdx]->queryResultBufferPtr->getCudaHandlePreQuery();

  return rtn;
#else
  CHECK(false) << "Cuda is not activated. Cannot get cuda handle.";
#endif  // HAVE_CUDA
}

void QueryRenderManager::setCudaBufferDataLayout(size_t gpuIdx,
                                                 size_t offsetBytes,
                                                 size_t numUsedBytes,
                                                 const QueryDataLayoutShPtr& vertLayoutPtr) {
  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "Cannot set cuda handle results for gpu index " + std::to_string(gpuIdx) + ". There are only " +
                        std::to_string(inOrder.size()) + " gpus available.");

  RUNTIME_EX_ASSERT(vertLayoutPtr, "A valid data layout is required");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_bufferMtx);

  inOrder[gpuIdx]->queryResultBufferPtr->setQueryDataLayout(
      _gpuCache->supportedExtensions, vertLayoutPtr, numUsedBytes, offsetBytes);
}

void QueryRenderManager::setCudaHandleUsedBytes(size_t gpuIdx,
                                                size_t numUsedBytes,
                                                const QueryDataLayoutShPtr& vertLayoutPtr) {
#ifdef HAVE_CUDA
  // TODO(croot): deprecate the vertLayoutPtr argument
  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "Cannot set cuda handle results for gpu index " + std::to_string(gpuIdx) + ". There are only " +
                        std::to_string(inOrder.size()) + " gpus available.");

  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_bufferMtx);

  ActiveRendererGuard activeRendererGuard(inOrder[gpuIdx].get());
  inOrder[gpuIdx]->queryResultBufferPtr->updatePostQuery(numUsedBytes);

  if (vertLayoutPtr) {
    inOrder[gpuIdx]->queryResultBufferPtr->setQueryDataLayout(
        _gpuCache->supportedExtensions, vertLayoutPtr, numUsedBytes);
  } else if (!numUsedBytes) {
    inOrder[gpuIdx]->queryResultBufferPtr->deleteAllQueryDataLayouts();
  }
#else
  CHECK(false) << "Cuda is not activated. Cannot set cuda handle bytes.";
#endif  // HAVE_CUDA
}

int QueryRenderManager::getPolyDataBufferAlignmentBytes(const size_t gpuIdx) const {
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::getPolyDataBufferAlignmentBytes(): Invalid gpu index " +
                        std::to_string(gpuIdx) + ". There are only " + std::to_string(inOrder.size()) +
                        " gpus available.");

  ActiveRendererGuard activeRendererGuard(inOrder[gpuIdx].get());

  return ::Rendering::GL::Resources::GLShaderBlockLayout::getNumAlignmentBytes();
}

bool QueryRenderManager::hasPolyTableGpuCache(const std::string& polyTableName, const size_t gpuIdx) const {
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::hasPolyTableGpuCache(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");

  return _gpuCache->hasPolyTableGpuCache(polyTableName, inOrder[gpuIdx]->gpuId);
}

bool QueryRenderManager::hasPolyTableGpuCache(const std::string& polyTableName,
                                              const std::string& sqlStr,
                                              const size_t gpuIdx) const {
  std::lock_guard<std::mutex> render_lock(_polyMtx);
  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::hasPolyTableGpuCache(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");
  return _gpuCache->hasPolyTableGpuCache(polyTableName, sqlStr, inOrder[gpuIdx]->gpuId);
}

PolyTableByteData QueryRenderManager::getPolyTableCacheByteInfo(const std::string& polyTableName,
                                                                const size_t gpuIdx) const {
  std::lock_guard<std::mutex> render_lock(_polyMtx);
  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::getPolyTableCacheByteInfo(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");

  auto itr = _gpuCache->polyCacheMap.find(polyTableName);
  RUNTIME_EX_ASSERT(itr != _gpuCache->polyCacheMap.end(),
                    "Cannot get poly table cache info for poly table " + polyTableName + ". The cache does not exist.");

  return itr->second.second.getPolyBufferByteData(inOrder[gpuIdx]->gpuId);
}

PolyTableDataInfo QueryRenderManager::getPolyTableCacheDataInfo(const std::string& polyTableName,
                                                                const size_t gpuIdx) const {
  std::lock_guard<std::mutex> render_lock(_polyMtx);
  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::getPolyTableCacheDataInfo(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");

  auto itr = _gpuCache->polyCacheMap.find(polyTableName);
  RUNTIME_EX_ASSERT(itr != _gpuCache->polyCacheMap.end(),
                    "Cannot get poly table cache info for poly table " + polyTableName + ". The cache does not exist.");

  return itr->second.second.getPolyBufferData(inOrder[gpuIdx]->gpuId);
}

void QueryRenderManager::createPolyTableCache(const std::string& polyTableName,
                                              const size_t gpuIdx,
                                              const PolyTableByteData& initTableData,
                                              const QueryDataLayoutShPtr& vertLayoutPtr,
                                              const PolyRowDataShPtr& rowDataPtr) {
  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::createPolyTableCache(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");

  ActiveRendererGuard activeRendererGuard;

  try {
    auto itr = _gpuCache->polyCacheMap.find(polyTableName);

    if (itr == _gpuCache->polyCacheMap.end()) {
      auto insertItr = _gpuCache->polyCacheMap.emplace(std::piecewise_construct,
                                                       std::forward_as_tuple(polyTableName),
                                                       std::forward_as_tuple(_gpuCache->buildSqlHash(""), _gpuCache));
      itr = insertItr.first;
    }

    itr->second.second.allocBuffers(inOrder[gpuIdx]->gpuId, initTableData, vertLayoutPtr, rowDataPtr);
  } catch (const ::Rendering::RenderError& e) {
    _gpuCache->polyCacheMap.erase(polyTableName);
    throw e;
  }
}

void QueryRenderManager::updatePolyTableCache(const std::string& polyTableName,
                                              const size_t gpuIdx,
                                              const PolyTableByteData& initTableData,
                                              const QueryDataLayoutShPtr& vertLayoutPtr,
                                              const PolyRowDataShPtr& rowDataPtr) {
  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::updatePolyTableCache(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");

  ActiveRendererGuard activeRendererGuard;

  auto itr = _gpuCache->polyCacheMap.find(polyTableName);

  RUNTIME_EX_ASSERT(itr != _gpuCache->polyCacheMap.end(),
                    "Cannot get poly table cache info for poly table " + polyTableName + ". The cache does not exist.");

  try {
    itr->second.second.allocBuffers(inOrder[gpuIdx]->gpuId, initTableData, vertLayoutPtr, rowDataPtr);
  } catch (const ::Rendering::RenderError& e) {
    _gpuCache->polyCacheMap.erase(polyTableName);
    throw e;
  }
}

void QueryRenderManager::deletePolyTableCache(const std::string& polyTableName) {
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  ActiveRendererGuard activeRendererGuard;
  _gpuCache->polyCacheMap.erase(polyTableName);

  // TODO(croot): what do we do about any query renderer data tables that might be
  // referencing removed caches? Right now the buffers will be kept around until
  // all those query renderers are configured with different data table references
}

void QueryRenderManager::deleteAllPolyTableCaches() {
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  ActiveRendererGuard activeRendererGuard;
  _gpuCache->polyCacheMap.clear();

  // TODO(croot): see TODO in deletePolyTableCache()
}

PolyCudaHandles QueryRenderManager::getPolyTableCudaHandles(const std::string& polyTableName,
                                                            const size_t gpuIdx,
                                                            const PolyTableByteData* initTableData) {
#ifdef HAVE_CUDA
  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  auto itr = _gpuCache->polyCacheMap.find(polyTableName);
  RUNTIME_EX_ASSERT(
      itr != _gpuCache->polyCacheMap.end(),
      "Cannot get cuda handles for poly table " + polyTableName + ". A cache for the table does not exist.");

  // TODO(croot): add a non-cached approach

  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::getPolyTableCudaHandles(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");

  ActiveRendererGuard activeRendererGuard(inOrder[gpuIdx].get());

  if (initTableData) {
    try {
      itr->second.second.allocBuffers(inOrder[gpuIdx]->gpuId, *initTableData);
    } catch (const ::Rendering::RenderError& e) {
      _gpuCache->polyCacheMap.erase(polyTableName);
      throw e;
    }
  }

  // reset the query str
  itr->second.first = _gpuCache->buildSqlHash("");
  return itr->second.second.getCudaHandlesPreQuery(inOrder[gpuIdx]->gpuId);
#else
  CHECK(false) << "Cuda is not activated. Cannot get cuda handle.";
#endif  // HAVE_CUDA
}

void QueryRenderManager::setPolyTableReadyForRender(const std::string& polyTableName,
                                                    const std::string& queryStr,
                                                    size_t gpuIdx,
                                                    const QueryDataLayoutShPtr& uniformLayoutPtr,
                                                    const PolyRowDataShPtr& rowDataPtr,
                                                    const QueryDataLayoutShPtr& vertLayoutPtr) {
#ifdef HAVE_CUDA
  // TODO(croot): Is the lock necessary here? Or should we lock on a per-gpu basis?
  std::lock_guard<std::mutex> render_lock(_polyMtx);

  auto itr = _gpuCache->polyCacheMap.find(polyTableName);
  RUNTIME_EX_ASSERT(
      itr != _gpuCache->polyCacheMap.end(),
      "Cannot get cuda handles for poly table " + polyTableName + ". A cache for the table does not exist.");

  RootPerGpuDataMap_in_order& inOrder = _gpuCache->perGpuData->get<inorder>();
  RUNTIME_EX_ASSERT(gpuIdx < inOrder.size(),
                    "QueryRenderManager::setPolyTableReadyForRender(): Invalid gpu index " + std::to_string(gpuIdx) +
                        ". There are only " + std::to_string(inOrder.size()) + " gpus available.");

  ActiveRendererGuard activeRendererGuard(inOrder[gpuIdx].get());
  itr->second.second.updatePostQuery(inOrder[gpuIdx]->gpuId, vertLayoutPtr, uniformLayoutPtr, rowDataPtr);
  itr->second.first = _gpuCache->buildSqlHash(queryStr);
#else
  CHECK(false) << "Cuda is not activated. Cannot set cuda handle bytes.";
#endif  // HAVE_CUDA
}

void QueryRenderManager::_configureRenderInternal(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr) {
  _activeItr->renderer->setJSONDocument(jsonDocumentPtr, false);
}

void QueryRenderManager::_configureRenderInternal(const std::string& jsonDocumentStr) {
  _activeItr->renderer->setJSONConfig(jsonDocumentStr, false);
}

void QueryRenderManager::configureRender(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                                         Executor* executor) {
  // DEPRECATED
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "ConfigureRender: There is no active user/widget id. Must set a user/widget id active before "
                    "configuring the render.");

  ActiveRendererGuard activeRendererGuard;

  // need to update the data layout of the query result buffer before building up
  // from the json obj
  if (executor) {
    _activeItr->renderer->updateResultsPostQuery(executor);
  }

  return _configureRenderInternal(jsonDocumentPtr);
}

size_t QueryRenderManager::getNumGpus() const {
  return _gpuCache->perGpuData->size();
}

std::vector<GpuId> QueryRenderManager::getAllGpuIds() const {
  std::vector<GpuId> rtn;
  for (auto itr : *(_gpuCache->perGpuData)) {
    rtn.push_back(itr->gpuId);
  }

  return rtn;
}

void QueryRenderManager::_renderInternal() {
  ActiveRendererGuard activeRendererGuard;

  _activeItr->renderer->render();
  _updateActiveLastRenderTime();
}

void QueryRenderManager::render() {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "render(): There is no active user/widget id. Must set a user/widget id active before rendering.");

  _renderInternal();
}

PngData QueryRenderManager::_renderToPngInternal(int compressionLevel) {
  PngData rtn = _activeItr->renderer->renderToPng(compressionLevel);

  _updateActiveLastRenderTime();

  return rtn;
}

PngData QueryRenderManager::renderToPng(int compressionLevel) {
  // DEPRECATED
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  ActiveRendererGuard activeRendererGuard(nullptr, this);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "There is no active user/widget id. Must set a user/widget id active before rendering.");

  return _renderToPngInternal(compressionLevel);
}

std::tuple<std::string, int64_t, int64_t> QueryRenderManager::runRenderRequest(int userId,
                                                                               int widgetId,
                                                                               const std::string& jsonStr,
                                                                               Executor* executor,
                                                                               RenderInfo* renderInfo,
                                                                               QueryExecCB queryExecFunc,
                                                                               int compressionLevel,
                                                                               bool doHitTest,
                                                                               bool doDepthTest) {
  auto renderTimerPtr = std::make_shared<RenderQueryExecuteTimer>();
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  ActiveRendererGuard activeRendererGuard;

  auto clock_begin = timer_start();
  if (!_hasUserWidgetInternal(userId, widgetId)) {
    _addUserWidgetInternal(userId, widgetId, doHitTest, doDepthTest);
  }

  _setActiveUserWidgetInternal(userId, widgetId);

  _activeItr->renderer->setQueryExecutionParams(executor, queryExecFunc, renderTimerPtr);

  try {
    _configureRenderInternal(jsonStr);

    CHECK(renderInfo);

    if (renderInfo->render_allocator_map_ptr) {
      // unmap any used buffers from cuda back to opengl
      renderInfo->render_allocator_map_ptr->prepForRendering(nullptr);
    }

    const auto pngdata = _renderToPngInternal(compressionLevel);

    int64_t render_time_ms = timer_stop(clock_begin);

    // Note: this time includes query queue time

    // TODO(croot): perhasp the best way to measure this time is to return
    // the time spent in the callback function and return that
    // as part of the _configureRenderInternal func or something
    // and subtract that from the total time here.

    _activeItr->renderer->unsetQueryExecutionParams();

    auto executionTime = renderTimerPtr->execution_time_ms;
    auto renderTime = render_time_ms - renderTimerPtr->execution_time_ms - renderTimerPtr->queue_time_ms;

    return std::make_tuple(std::string(pngdata.pngDataPtr.get(), pngdata.pngSize), executionTime, renderTime);
  } catch (const std::exception& e) {
    _activeItr->renderer->unsetQueryExecutionParams();
    throw std::runtime_error(e.what());
  }
}

std::tuple<int32_t, int64_t, std::string> QueryRenderManager::getIdAt(size_t x, size_t y, size_t pixelRadius) {
  std::lock_guard<std::mutex> render_lock(_renderMtx);

  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "getIdAt(): There is no active user/widget id. Must set an active user/widget id before "
                    "requesting pixel data.");

  ActiveRendererGuard activeRendererGuard;

  decltype(TableIdRowIdPair::first) tableId;
  decltype(TableIdRowIdPair::second) rowId;
  std::tie(tableId, rowId) = _activeItr->renderer->getIdAt(x, y, pixelRadius);

  // ids go from 0 to numitems-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
  int32_t rtnTableId = tableId - 1;
  auto rtnRowId = static_cast<int64_t>(rowId) - 1;
  std::string vega_table_name = _activeItr->renderer->getVegaTableNameWithTableId(rtnTableId);
  _updateActiveLastRenderTime();

  return std::make_tuple(rtnTableId, rtnRowId, vega_table_name);
}

bool QueryRenderManager::isPolyQueryCache(const TableId tableId) const {
  std::lock_guard<std::mutex> render_lock(_renderMtx);
  RUNTIME_EX_ASSERT(_activeItr != _rendererMap.end(),
                    "getIdAt(): There is no active user/widget id. Must set an active user/widget id before "
                    "requesting table info.");

  return _activeItr->renderer->getVegaTableTypeWithTableId(tableId) == QueryDataTableBaseType::POLY;
}

std::string QueryRenderManager::getQueryForQueryCache(const TableId tableId) const {
  return _gpuCache->renderQueryCacheMap.getQueryForCache(tableId);
}

std::pair<TableId, std::string> QueryRenderManager::getPrimaryQueryCacheTableInfo(const TableId tableId) const {
  return _gpuCache->renderQueryCacheMap.getQueryCachePrimaryTableInfo(tableId);
}

std::pair<const ResultRows*, const std::vector<TargetMetaInfo>*> QueryRenderManager::getQueryCacheResults(
    const TableId tableId) const {
  return _gpuCache->renderQueryCacheMap.getQueryCacheResults(tableId);
}

}  // namespace QueryRenderer
