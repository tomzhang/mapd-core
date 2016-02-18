#include "QueryRenderCompositor.h"
#ifdef MAPDGL_EGL
#include "egl/EglQueryRenderCompositorImpl.h"
#endif  // MAPDGL_EGL
#include <Rendering/RenderError.h>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

QueryRenderCompositor::QueryRenderCompositor(QueryRenderer* prnt,
                                             GLRenderer* renderer,
                                             size_t width,
                                             size_t height,
                                             size_t numSamples,
                                             bool doHitTest,
                                             bool doDepthTest)
    : _queryRenderer(prnt),
#ifdef MAPDGL_EGL
      _implPtr(new Impl::EGL::EglQueryRenderCompositorImpl(renderer, width, height, numSamples, doHitTest, doDepthTest))
#endif
{
}

QueryRenderCompositor::~QueryRenderCompositor() {
}

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

void QueryRenderCompositor::resize(size_t width, size_t height) {
  CHECK(_implPtr);
  _implPtr->resize(width, height);
}

void QueryRenderCompositor::render() {
  CHECK(_implPtr);
  _implPtr->render(_queryRenderer);
}

::Rendering::GL::Resources::GLTexture2dShPtr QueryRenderCompositor::createFboTexture2d(
    ::Rendering::GL::GLRenderer* renderer,
    FboColorBuffer texType) {
  GLTexture2dShPtr rtn = _implPtr->createFboTexture2d(renderer, texType);
  _compositeTextures.insert(rtn);
  return rtn;
}

::Rendering::GL::Resources::GLRenderbufferShPtr QueryRenderCompositor::createFboRenderbuffer(GLRenderer* renderer,
                                                                                             FboRenderBuffer rboType) {
  GLRenderbufferShPtr rtn = _implPtr->createFboRenderbuffer(renderer, rboType);
  _compositeRbos.insert(rtn);
  return rtn;
}

void QueryRenderCompositor::deleteFboTexture2d(const GLTexture2dShPtr& texture2dPtr) {
}

void QueryRenderCompositor::deleteFboRenderbuffer(const GLRenderbufferShPtr& renderbufferPtr) {
}

}  // namespace QueryRenderer
