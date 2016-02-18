#include "QueryFramebuffer.h"
#include "QueryRenderCompositor.h"
#include <Rendering/RenderError.h>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

QueryRenderCompositor::QueryRenderCompositor(GLRenderer* renderer,
                                             size_t width,
                                             size_t height,
                                             size_t numSamples,
                                             bool doHitTest,
                                             bool doDepthTest)
    : _framebufferPtr(nullptr) {
  _framebufferPtr.reset(new QueryFramebuffer(renderer, width, height, doHitTest, doDepthTest));
}

QueryRenderCompositor::~QueryRenderCompositor() {
}

size_t QueryRenderCompositor::getWidth() {
  CHECK(_framebufferPtr);
  return _framebufferPtr->getWidth();
}

size_t QueryRenderCompositor::getHeight() {
  CHECK(_framebufferPtr);
  return _framebufferPtr->getHeight();
}

// size_t QueryRenderCompositor::getNumSamples() {
//   CHECK(_framebufferPtr);
//   return _framebufferPtr->getNumSamples();
// }

::Rendering::GL::Resources::GLTexture2dShPtr QueryRenderCompositor::createFboTexture2d(
    ::Rendering::GL::GLRenderer* renderer,
    FboColorBuffer texType) {
  GLTexture2dShPtr rtn = _createFboTexture2dImpl(renderer, texType);
  _compositeTextures.insert(rtn);
  return rtn;
}

::Rendering::GL::Resources::GLRenderbufferShPtr QueryRenderCompositor::createFboRenderbuffer(GLRenderer* renderer,
                                                                                             FboRenderBuffer rboType) {
  GLRenderbufferShPtr rtn = _createFboRenderbufferImpl(renderer, rboType);
  return rtn;
}

void QueryRenderCompositor::deleteFboTexture2d(const GLTexture2dShPtr& texture2dPtr) {
}

void QueryRenderCompositor::deleteFboRenderbuffer(const GLRenderbufferShPtr& renderbufferPtr) {
}

}  // namespace QueryRenderer
