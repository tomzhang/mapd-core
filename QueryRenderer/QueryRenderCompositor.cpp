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

QueryRenderCompositor::QueryRenderCompositor(QueryRenderer* prnt,
                                             ::Rendering::RendererShPtr& rendererPtr,
                                             size_t width,
                                             size_t height,
                                             size_t numSamples,
                                             bool doHitTest,
                                             bool doDepthTest)
    :
#ifdef MAPDGL_GLX
      _implPtr(new Impl::GLX::GlxQueryRenderCompositorImpl(prnt,
                                                           rendererPtr,
                                                           width,
                                                           height,
                                                           numSamples,
                                                           doHitTest,
                                                           doDepthTest)),
#elif MAPDGL_EGL
      _implPtr(new Impl::EGL::EglQueryRenderCompositorImpl(prnt,
                                                           rendererPtr,
                                                           width,
                                                           height,
                                                           numSamples,
                                                           doHitTest,
                                                           doDepthTest)),
#endif  // MAPDGL_GLX
      _queryRenderer(prnt) {
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

void QueryRenderCompositor::resize(size_t width, size_t height) {
  CHECK(_implPtr);
  _implPtr->resize(width, height);
}

void QueryRenderCompositor::render() {
  CHECK(_implPtr);
  _implPtr->render(_queryRenderer);
}

std::shared_ptr<unsigned char> QueryRenderCompositor::readColorBuffer(size_t startx,
                                                                      size_t starty,
                                                                      int width,
                                                                      int height) {
  CHECK(_implPtr);
  return _implPtr->readColorBuffer(startx, starty, width, height);
}

std::shared_ptr<unsigned int> QueryRenderCompositor::readIdBuffer(size_t startx, size_t starty, int width, int height) {
  CHECK(_implPtr);
  return _implPtr->readIdBuffer(startx, starty, width, height);
}

::Rendering::GL::Resources::GLTexture2dShPtr QueryRenderCompositor::createFboTexture2d(
    ::Rendering::GL::GLRenderer* renderer,
    FboColorBuffer texType) {
  GLTexture2dShPtr rtn = _implPtr->createFboTexture2d(renderer, texType);
  _compositeTextures.insert({rtn.get(), rtn});
  return rtn;
}

::Rendering::GL::Resources::GLRenderbufferShPtr QueryRenderCompositor::createFboRenderbuffer(GLRenderer* renderer,
                                                                                             FboRenderBuffer rboType) {
  GLRenderbufferShPtr rtn = _implPtr->createFboRenderbuffer(renderer, rboType);
  _compositeRbos.insert({rtn.get(), rtn});
  return rtn;
}

void QueryRenderCompositor::addQueryFramebuffer(QueryFramebuffer* fbo) {
  ::Rendering::GL::Resources::GLTexture2dShPtr rgbaTex = fbo->getColorTexture2d(FboColorBuffer::COLOR_BUFFER);

  CHECK(_compositeTextures.find(rgbaTex.get()) == _compositeTextures.end());
  _implPtr->addFboTexture2d(rgbaTex, FboColorBuffer::COLOR_BUFFER);
  _compositeTextures.insert({rgbaTex.get(), rgbaTex});

  if (fbo->doHitTest()) {
    ::Rendering::GL::Resources::GLTexture2dShPtr idTex = fbo->getColorTexture2d(FboColorBuffer::ID_BUFFER);
    CHECK(_compositeTextures.find(idTex.get()) == _compositeTextures.end());
    _implPtr->addFboTexture2d(idTex, FboColorBuffer::ID_BUFFER);
    _compositeTextures.insert({idTex.get(), idTex});
  }

  if (fbo->doDepthTest()) {
    // TODO(croot): fill this out
    // ::Rendering::GL::Resources::GLRenderbufferShPtr rbo = fbo->getColorTexture2d(FboRenderBuffer::DEPTH_BUFFER);
    // _implPtr->addFboRenderbuffer(rbo, FboRenderBuffer::DEPTH_BUFFER);
    // _compositeRbos.insert({rbo.get(), rbo});
  }
}

void QueryRenderCompositor::addQueryFramebuffer(const QueryFramebufferShPtr& fbo) {
  addQueryFramebuffer(fbo.get());
}

void QueryRenderCompositor::addQueryFramebuffer(const QueryFramebufferUqPtr& fbo) {
  addQueryFramebuffer(fbo.get());
}

void QueryRenderCompositor::deleteFboTexture2d(GLTexture2d* texture2dPtr) {
  CHECK(_implPtr);
  _implPtr->deleteFboTexture2d(texture2dPtr);

  auto itr = _compositeTextures.find(texture2dPtr);
  if (itr != _compositeTextures.end()) {
    _compositeTextures.erase(itr);
  }
}

void QueryRenderCompositor::deleteFboRenderbuffer(GLRenderbuffer* renderbufferPtr) {
  CHECK(_implPtr);
  _implPtr->deleteFboRenderbuffer(renderbufferPtr);

  auto itr = _compositeRbos.find(renderbufferPtr);
  if (itr != _compositeRbos.end()) {
    _compositeRbos.erase(itr);
  }
}

void QueryRenderCompositor::cleanupUnusedFbos() {
  std::vector<GLTexture2d*> unusedTextures;
  std::vector<GLRenderbuffer*> unusedRbos;

  for (auto& itr : _compositeTextures) {
    if (itr.second.expired()) {
      unusedTextures.push_back(itr.first);
    }
  }

  for (auto& itr : _compositeRbos) {
    if (itr.second.expired()) {
      unusedRbos.push_back(itr.first);
    }
  }

  for (auto item : unusedTextures) {
    deleteFboTexture2d(item);
  }

  for (auto item : unusedRbos) {
    deleteFboRenderbuffer(item);
  }
}

}  // namespace QueryRenderer
