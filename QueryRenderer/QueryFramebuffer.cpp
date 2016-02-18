#include "QueryFramebuffer.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>

// #include "MapDGL.h"
// #include "QueryRendererError.h"
// #include "QueryFramebuffer.h"
// #include <iostream>
// #include <stdexcept>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;
using ::Rendering::GL::Resources::FboBind;
using ::Rendering::GL::Resources::GLFramebufferAttachmentMap;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

QueryFramebuffer::QueryFramebuffer(GLRenderer* renderer, int width, int height, bool doHitTest, bool doDepthTest)
    : _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _rgbaTex(nullptr),
      _idTex(nullptr),
      _rbo(nullptr),
      _fbo(nullptr) {
  _init(renderer, width, height);
}

QueryFramebuffer::~QueryFramebuffer() {
}

void QueryFramebuffer::_init(::Rendering::GL::GLRenderer* renderer, int width, int height) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _rgbaTex = createFboTexture2d(rsrcMgr, FboColorBuffer::COLOR_BUFFER, width, height);

  GLFramebufferAttachmentMap attachments({{GL_COLOR_ATTACHMENT0, _rgbaTex}});

  if (_doHitTest) {
    _idTex = createFboTexture2d(rsrcMgr, FboColorBuffer::ID_BUFFER, width, height);
    attachments.insert({GL_COLOR_ATTACHMENT1, _idTex});
  }

  if (_doDepthTest) {
    _rbo = createFboRenderbuffer(rsrcMgr, FboRenderBuffer::DEPTH_BUFFER, width, height);
    attachments.insert({GL_DEPTH_ATTACHMENT, _rbo});
  }

  _fbo = rsrcMgr->createFramebuffer(attachments);
}

void QueryFramebuffer::resize(int width, int height) {
  CHECK(_fbo);
  _fbo->resize(width, height);
}

int QueryFramebuffer::getWidth() const {
  return _fbo->getWidth();
}

int QueryFramebuffer::getHeight() const {
  return _fbo->getHeight();
}

GLuint QueryFramebuffer::getId(FboColorBuffer buffer) {
  switch (buffer) {
    case FboColorBuffer::COLOR_BUFFER:
      return (_rgbaTex ? _rgbaTex->getId() : 0);
    case FboColorBuffer::ID_BUFFER:
      return (_idTex ? _idTex->getId() : 0);
    default:
      return 0;
  }
  return 0;
}

GLuint QueryFramebuffer::getId(FboRenderBuffer buffer) {
  switch (buffer) {
    case FboRenderBuffer::DEPTH_BUFFER:
      return (_rbo ? _rbo->getId() : 0);
    default:
      return 0;
  }

  return 0;
}

void QueryFramebuffer::bindToRenderer(GLRenderer* renderer, FboBind bindType) {
  renderer->bindFramebuffer(bindType, _fbo);
}

std::shared_ptr<unsigned char> QueryFramebuffer::readColorBuffer(size_t startx,
                                                                 size_t starty,
                                                                 size_t width,
                                                                 size_t height) {
  std::shared_ptr<unsigned char> pixels(new unsigned char[width * height * 4], std::default_delete<unsigned char[]>());
  unsigned char* rawPixels = pixels.get();

  _fbo->readPixels(GL_COLOR_ATTACHMENT0, startx, starty, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rawPixels);

  return pixels;
}

GLTexture2dShPtr QueryFramebuffer::createFboTexture2d(GLResourceManagerShPtr& rsrcMgr,
                                                      FboColorBuffer texType,
                                                      size_t width,
                                                      size_t height) {
  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      return rsrcMgr->createTexture2d(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
    case FboColorBuffer::ID_BUFFER:
      return rsrcMgr->createTexture2d(width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
    default:
      CHECK(false);
  }

  return nullptr;
}

GLRenderbufferShPtr QueryFramebuffer::createFboRenderbuffer(GLResourceManagerShPtr& rsrcMgr,
                                                            FboRenderBuffer rboType,
                                                            size_t width,
                                                            size_t height) {
  switch (rboType) {
    case FboRenderBuffer::DEPTH_BUFFER:
      return rsrcMgr->createRenderbuffer(width, height, GL_DEPTH_COMPONENT);
    default:
      CHECK(false);
  }

  return nullptr;
}

}  // namespace QueryRenderer
