#include "PerGpuData.h"
#include "Rendering/QueryFramebuffer.h"
#include "Rendering/QueryRenderCompositor.h"
#include "Rendering/QueryIdMapPboPool.h"

#include <Rendering/Renderer/GL/GLRenderer.h>

namespace QueryRenderer {

RootPerGpuData::RootPerGpuData(GpuId gpuId)
    : gpuId(gpuId),
      queryResultBufferPtr(nullptr),
      windowPtr(nullptr),
      rendererPtr(nullptr),
      msFramebufferPtr(nullptr),
      blitFramebufferPtr(nullptr),
      compositorPtr(nullptr),
      pboPoolPtr(nullptr) {
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

void RootPerGpuData::resize(size_t width, size_t height) {
  CHECK(msFramebufferPtr);

  width = std::max(width, msFramebufferPtr->getWidth());
  height = std::max(height, msFramebufferPtr->getHeight());
  msFramebufferPtr->resize(width, height);

  ::Rendering::GL::GLRenderer* prevRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
  ::Rendering::Window* prevWindow = ::Rendering::GL::GLRenderer::getCurrentThreadWindow();
  bool reset = false;

  if (compositorPtr && (compositorPtr->getWidth() < width || compositorPtr->getHeight() < height)) {
    ::Rendering::GL::GLRenderer* renderer = compositorPtr->getGLRenderer();
    CHECK(renderer);
    if (renderer != prevRenderer) {
      reset = true;
      renderer->makeActiveOnCurrentThread();
    }
    compositorPtr->resize(width, height);
  }

  if (blitFramebufferPtr && (blitFramebufferPtr->getWidth() < width || blitFramebufferPtr->getHeight() < height)) {
    ::Rendering::GL::GLRenderer* renderer = blitFramebufferPtr->getGLRenderer();
    CHECK(renderer);
    if (renderer != prevRenderer) {
      reset = true;
      renderer->makeActiveOnCurrentThread();
    }
    blitFramebufferPtr->resize(width, height);
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

QueryIdMapPixelBufferShPtr RootPerGpuData::getInactiveIdMapPbo(size_t width, size_t height) {
  CHECK(pboPoolPtr);
  return pboPoolPtr->getInactiveIdMapPbo(width, height);
}

void RootPerGpuData::setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo) {
  CHECK(pboPoolPtr);
  pboPoolPtr->setIdMapPboInactive(pbo);
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

QueryFramebufferShPtr& BasePerGpuData::getBlitFramebuffer() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getBlitFramebuffer();
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

QueryIdMapPixelBufferShPtr BasePerGpuData::getInactiveIdMapPbo(size_t width, size_t height) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getInactiveIdMapPbo(width, height);
}

void BasePerGpuData::setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->setIdMapPboInactive(pbo);
  }
}

}  // namespace QueryRenderer
