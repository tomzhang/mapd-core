#include "PerGpuData.h"
#include "Rendering/QueryFramebuffer.h"
#include "Rendering/QueryRenderCompositor.h"
#include "Rendering/QueryIdMapPboPool.h"
#include "Rendering/QueryAccumTxPool.h"
#include "Rendering/QueryRenderSMAAPass.h"
#include "Rendering/DistributedRenderBufferCompositor.h"
#include <Rendering/Renderer/GL/GLRenderer.h>

namespace QueryRenderer {

RootPerGpuData::RootPerGpuData(GpuId gpuId) : gpuId(gpuId) {}

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

::Rendering::GL::GLRendererShPtr RootPerGpuData::getGLRendererShPtr() {
  return std::dynamic_pointer_cast<::Rendering::GL::GLRenderer>(rendererPtr);
}

void RootPerGpuData::resize(size_t width, size_t height, bool resizeCompositor) {
  CHECK(msFramebufferPtr);

  ::Rendering::GL::GLRenderer* prevRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
  ::Rendering::Window* prevWindow = ::Rendering::GL::GLRenderer::getCurrentThreadWindow();
  bool reset = false;

  auto setRenderer = [prevRenderer, &reset](::Rendering::GL::GLRenderer* renderer) {
    CHECK(renderer);
    if (!reset && renderer != prevRenderer) {
      reset = true;
      renderer->makeActiveOnCurrentThread();
    }
  };

  width = std::max(width, msFramebufferPtr->getWidth());
  height = std::max(height, msFramebufferPtr->getHeight());

  // TODO(croot): only resize the compositor if we're on the same gpu unless specified otherwise?

  // need to resize the compositor first
  if (resizeCompositor && compositorPtr && (compositorPtr->getWidth() < width || compositorPtr->getHeight() < height)) {
    setRenderer(compositorPtr->getGLRenderer());
    compositorPtr->resize(width, height);
  }

  msFramebufferPtr->resize(width, height);

  if (accumTxPoolPtr) {
    accumTxPoolPtr->resize(width, height);
  }

  if (aaFramebufferPtr && (aaFramebufferPtr->getWidth() < width || aaFramebufferPtr->getHeight() < height)) {
    setRenderer(aaFramebufferPtr->getGLRenderer());
    aaFramebufferPtr->resize(width, height);
  }

  if (smaaPassPtr && (smaaPassPtr->getWidth() < width || smaaPassPtr->getHeight() < height)) {
    setRenderer(smaaPassPtr->getGLRenderer());
    smaaPassPtr->resize(width, height);
  }

  if (distCompPtr && (distCompPtr->getWidth() < width || distCompPtr->getHeight() < height)) {
    setRenderer(distCompPtr->getGLRenderer());
    distCompPtr->resize(width, height);
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

QueryIdMapPixelBufferUIntWkPtr RootPerGpuData::getInactiveRowIdMapPbo(size_t width, size_t height) {
  CHECK(pboPoolUIntPtr);
  return pboPoolUIntPtr->getInactiveRsrc(width, height);
}

void RootPerGpuData::setRowIdMapPboInactive(QueryIdMapPixelBufferUIntWkPtr& pbo) {
  CHECK(pboPoolUIntPtr);
  pboPoolUIntPtr->setRsrcInactive(pbo);
}

QueryIdMapPixelBufferIntWkPtr RootPerGpuData::getInactiveTableIdMapPbo(size_t width, size_t height) {
  CHECK(pboPoolIntPtr);
  return pboPoolIntPtr->getInactiveRsrc(width, height);
}

void RootPerGpuData::setTableIdMapPboInactive(QueryIdMapPixelBufferIntWkPtr& pbo) {
  CHECK(pboPoolIntPtr);
  pboPoolIntPtr->setRsrcInactive(pbo);
}

QueryAccumTxPoolUqPtr& RootPerGpuData::getAccumTxPool() {
  if (!accumTxPoolPtr) {
    auto glRenderer = std::dynamic_pointer_cast<::Rendering::GL::GLRenderer>(rendererPtr);
    CHECK(glRenderer);
    accumTxPoolPtr.reset(new QueryAccumTxPool(glRenderer, compositorPtr));
  }

  return accumTxPoolPtr;
}

DistributedRenderBufferCompositorShPtr RootPerGpuData::getDistributedCompositorPtr(const bool supportsInt64) {
  if (!distCompPtr) {
    auto fbo = aaFramebufferPtr;
    if (!fbo) {
      fbo = msFramebufferPtr;
    }
    RUNTIME_EX_ASSERT(fbo && fbo->getNumSamples() == 1,
                      "Cannot create distributed render buffer compositor. Framebuffer to use for the compositor has " +
                          std::to_string(fbo->getNumSamples()) + " samples. It needs to be a 1 sample fbo.");
    // TODO(croot): pass thru the flag that indicates whether this gpu context supports
    // 64-bit ints
    distCompPtr.reset(new DistributedRenderBufferCompositor(fbo, supportsInt64));
  }
  return distCompPtr;
}

GpuId BasePerGpuData::getGpuId() const {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();

  RUNTIME_EX_ASSERT(qrmGpuDataShPtr, "Cannot get gpu id. Root gpu info has been deleted.");
  return qrmGpuDataShPtr->gpuId;
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

::Rendering::GL::GLRendererShPtr BasePerGpuData::getGLRendererShPtr() {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    return qrmGpuDataShPtr->getGLRendererShPtr();
  }

  return nullptr;
}

void BasePerGpuData::resize(size_t width, size_t height) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->resize(width, height);
  }
}

QueryFramebufferShPtr& BasePerGpuData::getRenderFramebuffer() {
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

QueryIdMapPixelBufferUIntWkPtr BasePerGpuData::getInactiveRowIdMapPbo(size_t width, size_t height) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getInactiveRowIdMapPbo(width, height);
}

void BasePerGpuData::setRowIdMapPboInactive(QueryIdMapPixelBufferUIntWkPtr& pbo) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  if (qrmGpuDataShPtr) {
    qrmGpuDataShPtr->setRowIdMapPboInactive(pbo);
  }
}

QueryIdMapPixelBufferIntWkPtr BasePerGpuData::getInactiveTableIdMapPbo(size_t width, size_t height) {
  RootPerGpuDataShPtr qrmGpuDataShPtr = rootPerGpuData.lock();
  CHECK(qrmGpuDataShPtr);
  return qrmGpuDataShPtr->getInactiveTableIdMapPbo(width, height);
}

void BasePerGpuData::setTableIdMapPboInactive(QueryIdMapPixelBufferIntWkPtr& pbo) {
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
