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
using ::Rendering::GL::Resources::GLFramebufferAttachmentMap;

QueryFramebuffer::QueryFramebuffer(GLRenderer* renderer, int width, int height, bool doHitTest, bool doDepthTest)
    : _rgbaTex(nullptr),
      _idTex(nullptr),
      _rbo(nullptr),
      _fbo(nullptr),
      _doHitTest(doHitTest),
      _doDepthTest(doDepthTest) {
  _init(renderer, width, height);
}

QueryFramebuffer::~QueryFramebuffer() {
}

void QueryFramebuffer::_init(::Rendering::GL::GLRenderer* renderer, int width, int height) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  _rgbaTex = rsrcMgr->createTexture2d(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);

  GLFramebufferAttachmentMap attachments({{GL_COLOR_ATTACHMENT0, _rgbaTex}});

  if (_doHitTest) {
    _idTex = rsrcMgr->createTexture2d(width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT);
    attachments.insert({GL_COLOR_ATTACHMENT1, _idTex});
  }

  if (_doDepthTest) {
    _rbo = rsrcMgr->createRenderbuffer(width, height, GL_DEPTH_COMPONENT);
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

// void QueryFramebuffer::bindToRenderer(BindType bindType) {
//   MAPD_CHECK_GL_ERROR(glBindFramebuffer(static_cast<int>(bindType), _fbo));
//   // _attachmentManager.enableAttachments();

//   // TODO: How do we properly deal with the state machine
//   if (_renderBuffers[DEPTH_BUFFER]) {
//     MAPD_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
//   }
// }

}  // namespace QueryRenderer
