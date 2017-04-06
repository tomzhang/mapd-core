#include "QueryRenderCompositor.h"
#ifdef MAPDGL_GLX
#include "glx/GlxQueryRenderCompositorImpl.h"
#elif MAPDGL_EGL
#include "egl/EglQueryRenderCompositorImpl.h"
#endif  // MAPDGL_EGL
#include <Rendering/RenderError.h>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::Resources::GLTexture2d;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbuffer;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

QueryRenderCompositor::QueryRenderCompositor(QueryRenderManager* prnt,
                                             ::Rendering::RendererShPtr& rendererPtr,
                                             const size_t width,
                                             const size_t height,
                                             const size_t numSamples,
                                             const bool doHitTest,
                                             const bool doDepthTest,
                                             const bool supportsInt64)
    :
#ifdef MAPDGL_GLX
      _implPtr(new Impl::GLX::GlxQueryRenderCompositorImpl(prnt,
                                                           rendererPtr,
                                                           width,
                                                           height,
                                                           numSamples,
                                                           doHitTest,
                                                           doDepthTest,
                                                           supportsInt64))
#elif MAPDGL_EGL
      _implPtr(new Impl::EGL::EglQueryRenderCompositorImpl(prnt,
                                                           rendererPtr,
                                                           width,
                                                           height,
                                                           numSamples,
                                                           doHitTest,
                                                           doDepthTest,
                                                           supportsInt64))
#else
      _implPtr(nullptr)
#endif  // MAPDGL_GLX
{
}

QueryRenderCompositor::~QueryRenderCompositor() {}

size_t QueryRenderCompositor::getWidth() {
  CHECK(_implPtr);
  return _implPtr->getWidth();
}

size_t QueryRenderCompositor::getHeight() {
  CHECK(_implPtr);
  return _implPtr->getHeight();
}

// size_t QueryRenderCompositor::getNumSamples() {
//   CHECK(_framebufferPtr);
//   return _framebufferPtr->getNumSamples();
// }

bool QueryRenderCompositor::doHitTest() {
  CHECK(_implPtr);
  return _implPtr->doHitTest();
}

bool QueryRenderCompositor::doDepthTest() {
  CHECK(_implPtr);
  return _implPtr->doDepthTest();
}

::Rendering::Renderer* QueryRenderCompositor::getRenderer() {
  CHECK(_implPtr);
  return _implPtr->getRenderer();
}

::Rendering::GL::GLRenderer* QueryRenderCompositor::getGLRenderer() {
  CHECK(_implPtr);
  return _implPtr->getGLRenderer();
}

void QueryRenderCompositor::resize(size_t width, size_t height) {
  CHECK(_implPtr);
  _implPtr->resize(width, height);
}

QueryFramebufferUqPtr& QueryRenderCompositor::render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) {
  CHECK(_implPtr);
  _implPtr->render(queryRenderer, usedGpus);
  return _implPtr->getFramebuffer();
}

std::shared_ptr<unsigned char> QueryRenderCompositor::readColorBuffer(size_t startx,
                                                                      size_t starty,
                                                                      int width,
                                                                      int height) {
  CHECK(_implPtr);
  return _implPtr->readColorBuffer(startx, starty, width, height);
}

std::shared_ptr<unsigned int> QueryRenderCompositor::readIdBuffer(size_t startx,
                                                                  size_t starty,
                                                                  int width,
                                                                  int height,
                                                                  const FboColorBuffer idBufferType) {
  CHECK(_implPtr);
  return _implPtr->readIdBuffer(startx, starty, width, height, idBufferType);
}

void QueryRenderCompositor::copyRowIdBufferToPbo(QueryIdMapPixelBufferUIntShPtr& pbo, const bool leastSignificantBits) {
  CHECK(_implPtr);
  return _implPtr->copyRowIdBufferToPbo(pbo, leastSignificantBits);
}

void QueryRenderCompositor::copyTableIdBufferToPbo(QueryIdMapPixelBufferIntShPtr& pbo) {
  CHECK(_implPtr);
  return _implPtr->copyTableIdBufferToPbo(pbo);
}

void QueryRenderCompositor::readIdBuffer(size_t startx, size_t starty, int width, int height, unsigned int* idBuffer) {
  CHECK(_implPtr);
  return _implPtr->readIdBuffer(startx, starty, width, height, idBuffer);
}

::Rendering::GL::Resources::GLTexture2dShPtr QueryRenderCompositor::createFboTexture2d(
    ::Rendering::GL::GLRenderer* renderer,
    FboColorBuffer texType) {
  GLTexture2dShPtr rtn = _implPtr->createFboTexture2d(renderer, texType);
  return rtn;
}

::Rendering::GL::Resources::GLRenderbufferShPtr QueryRenderCompositor::createFboRenderbuffer(GLRenderer* renderer,
                                                                                             FboRenderBuffer rboType) {
  GLRenderbufferShPtr rtn = _implPtr->createFboRenderbuffer(renderer, rboType);
  return rtn;
}

void QueryRenderCompositor::registerAccumulatorTexture(::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                                       size_t accumIdx,
                                                       size_t numTexturesInArray) {
  CHECK(_implPtr);
  return _implPtr->registerAccumulatorTexture(tex, accumIdx, numTexturesInArray);
}

void QueryRenderCompositor::unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                                         size_t accumIdx) {
  CHECK(_implPtr);
  return _implPtr->unregisterAccumulatorTexture(tex, accumIdx);
}

void QueryRenderCompositor::unregisterAllAccumulatorTextures() {
  CHECK(_implPtr);
  return _implPtr->unregisterAllAccumulatorTextures();
}

}  // namespace QueryRenderer
