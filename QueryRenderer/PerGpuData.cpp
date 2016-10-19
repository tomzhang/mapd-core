#include "PerGpuData.h"
#include "Rendering/QueryFramebuffer.h"
#include "Rendering/QueryRenderCompositor.h"
#include "Rendering/QueryIdMapPboPool.h"
#include "Rendering/QueryAccumTxPool.h"
#include "Rendering/QueryRenderSMAAPass.h"

#include <Rendering/Renderer/GL/GLRenderer.h>

namespace QueryRenderer {

RootPerGpuData::RootPerGpuData(GpuId gpuId) : gpuId(gpuId) {
}

RootPerGpuData::~RootPerGpuData() {
  // need to make active to properly destroy gpu resources
  // TODO(croot): reset to previously active renderer?
  makeActiveOnCurrentThread();
}

void RootPerGpuData::makeActiveOnCurrentThread() const {
  CHECK(windowPtr && rendererPtr);
  rendererPtr->makeActiveOnCurrentThread(windowPtr);
}

void RootPerGpuData::makeInactive() const {
  CHECK(rendererPtr);
  rendererPtr->makeInactive();
}

::Rendering::Renderer* RootPerGpuData::getRenderer() {
  if (rendererPtr) {
    return rendererPtr.get();
  }
  return nullptr;
}

::Rendering::GL::GLRenderer* RootPerGpuData::getGLRenderer() {
  if (rendererPtr) {
    ::Rendering::Renderer* renderer = rendererPtr.get();
    return dynamic_cast<::Rendering::GL::GLRenderer*>(renderer);
  }

  return nullptr;
}

void RootPerGpuData::resize(size_t width, size_t height, bool resizeCompositor) {
  CHECK(msFramebufferPtr);

  width = std::max(width, msFramebufferPtr->getWidth());
  height = std::max(height, msFramebufferPtr->getHeight());
  msFramebufferPtr->resize(width, height);

  if (accumTxPoolPtr) {
    accumTxPoolPtr->resize(width, height);
  }

  ::Rendering::GL::GLRenderer* prevRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
  ::Rendering::Window* prevWindow = ::Rendering::GL::GLRenderer::getCurrentThreadWindow();
  bool reset = false;

  // TODO(croot): only resize the compositor if we're on the same gpu unless specified otherwise?
  if (resizeCompositor && compositorPtr && (compositorPtr->getWidth() < width || compositorPtr->getHeight() < height)) {
    ::Rendering::GL::GLRenderer* renderer = compositorPtr->getGLRenderer();
    CHECK(renderer);
    if (renderer != prevRenderer) {
      reset = true;
      renderer->makeActiveOnCurrentThread();
      prevRenderer = renderer;
    }
    compositorPtr->resize(width, height);
  }

  if (aaFramebufferPtr && (aaFramebufferPtr->getWidth() < width || aaFramebufferPtr->getHeight() < height)) {
    ::Rendering::GL::GLRenderer* renderer = aaFramebufferPtr->getGLRenderer();
    CHECK(renderer);
    if (renderer != prevRenderer) {
      reset = true;
      renderer->makeActiveOnCurrentThread();
      prevRenderer = renderer;
    }
    aaFramebufferPtr->resize(width, height);
  }

  if (smaaPassPtr && (smaaPassPtr->getWidth() < width || smaaPassPtr->getHeight() < height)) {
    ::Rendering::GL::GLRenderer* renderer = smaaPassPtr->getGLRenderer();
    CHECK(renderer);
    if (renderer != prevRenderer) {
      reset = true;
      renderer->makeActiveOnCurrentThread();
    }
    smaaPassPtr->resize(width, height);
  }

  if (reset) {
    prevRenderer->makeActiveOnCurrentThread(prevWindow);
  }
}

GpuId RootPerGpuData::getCompositorGpuId() {
  RUNTIME_EX_ASSERT(compositorPtr != nullptr, "Cannot get the compositor gpu id. The compositor is uninitialized.");

  ::Rendering::GL::GLRenderer* renderer = compositorPtr->getGLRenderer();
  CHECK(renderer);
  return renderer->getGpuId();
}

QueryIdMapPixelBufferUIntShPtr RootPerGpuData::getInactiveRowIdMapPbo(size_t width, size_t height) {
  CHECK(pboPoolUIntPtr);
  return pboPoolUIntPtr->getInactiveIdMapPbo(width, height);
}

void RootPerGpuData::setRowIdMapPboInactive(QueryIdMapPixelBufferUIntShPtr& pbo) {
  CHECK(pboPoolUIntPtr);
  pboPoolUIntPtr->setIdMapPboInactive(pbo);
}

QueryIdMapPixelBufferIntShPtr RootPerGpuData::getInactiveTableIdMapPbo(size_t width, size_t height) {
  CHECK(pboPoolIntPtr);
  return pboPoolIntPtr->getInactiveIdMapPbo(width, height);
}

void RootPerGpuData::setTableIdMapPboInactive(QueryIdMapPixelBufferIntShPtr& pbo) {
  CHECK(pboPoolIntPtr);
  pboPoolIntPtr->setIdMapPboInactive(pbo);
}

QueryAccumTxPoolUqPtr& RootPerGpuData::getAccumTxPool() {
  if (!accumTxPoolPtr) {
    auto glRenderer = std::dynamic_pointer_cast<::Rendering::GL::GLRenderer>(rendererPtr);
    CHECK(glRenderer);
    accumTxPoolPtr.reset(new QueryAccumTxPool(glRenderer, compositorPtr));
  }

  return accumTxPoolPtr;
}

void BasePerGpuData::makeActiveOnCurrentThread() const {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->makeActiveOnCurrentThread();
  }
}

void BasePerGpuData::makeInactive() const {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->makeInactive();
  }
}

::Rendering::Renderer* BasePerGpuData::getRenderer() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    return qrmGpuDataShPtr->getRenderer();
  }

  return nullptr;
}

::Rendering::GL::GLRenderer* BasePerGpuData::getGLRenderer() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    return qrmGpuDataShPtr->getGLRenderer();
  }

  return nullptr;
}

void BasePerGpuData::resize(size_t width, size_t height) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->resize(width, height);
  }
}

QueryFramebufferUqPtr& BasePerGpuData::getRenderFramebuffer() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getRenderFramebuffer();
}

QueryFramebufferShPtr& BasePerGpuData::getAntiAliasingFramebuffer() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getAntiAliasingFramebuffer();
}

bool BasePerGpuData::hasCompositor() const {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  return (qrmGpuDataShPtr != nullptr && qrmGpuDataShPtr->hasCompositor());
}

std::shared_ptr<QueryRenderCompositor>& BasePerGpuData::getCompositor() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getCompositor();
}

GpuId BasePerGpuData::getCompositorGpuId() {
  auto compositorPtr = getCompositor();
  ::Rendering::GL::GLRenderer* renderer = compositorPtr->getGLRenderer();
  CHECK(renderer);
  return renderer->getGpuId();
}

QueryIdMapPixelBufferUIntShPtr BasePerGpuData::getInactiveRowIdMapPbo(size_t width, size_t height) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getInactiveRowIdMapPbo(width, height);
}

void BasePerGpuData::setRowIdMapPboInactive(QueryIdMapPixelBufferUIntShPtr& pbo) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->setRowIdMapPboInactive(pbo);
  }
}

QueryIdMapPixelBufferIntShPtr BasePerGpuData::getInactiveTableIdMapPbo(size_t width, size_t height) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getInactiveTableIdMapPbo(width, height);
}

void BasePerGpuData::setTableIdMapPboInactive(QueryIdMapPixelBufferIntShPtr& pbo) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->setTableIdMapPboInactive(pbo);
  }
}

bool BasePerGpuData::hasAccumTxPool() const {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  return (qrmGpuDataShPtr != nullptr && qrmGpuDataShPtr->hasAccumTxPool());
}

QueryAccumTxPoolUqPtr& BasePerGpuData::getAccumTxPool() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getAccumTxPool();
}

}  // namespace QueryRenderer
