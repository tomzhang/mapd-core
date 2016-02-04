#include "GLWindow.h"
#include "MapDGL.h"
// #include "GLRenderer.h"
#include "Resources/GLFramebuffer.h"
#include "../../Settings/WindowSettings.h"
#include <iostream>

namespace Rendering {

using Settings::IntSetting;

namespace GL {

GLWindow::GLWindow(const WindowSettings& windowSettings, const WindowShPtr& prnt)
    : Window(RendererType::OPENGL, windowSettings, prnt) {
}

// GLWindow::GLWindow(const WindowSettings& windowSettings, const RendererShPtr& rendererPtr, const WindowShPtr& prnt)
//     : Window(windowSettings, rendererPtr, prnt) {
// }

// GLWindow::GLWindow(const WindowShPtr& prnt) : Window(prnt) {
// }

GLWindow::~GLWindow() {
  std::cerr << "CROOT GLWindow destructor" << std::endl;
}

// TODO(croot): what if we want to blit a color, depth and/or stencil buffer?
void GLWindow::blitFramebuffer(const Resources::GLFramebufferShPtr& fbo,
                               GLenum attachment,
                               const Viewport& srcFboBounds,
                               const Viewport& dstFboBounds,
                               GLenum filter) {
  blitFramebuffer(fbo,
                  attachment,
                  srcFboBounds.getXPos(),
                  srcFboBounds.getYPos(),
                  srcFboBounds.getWidth(),
                  srcFboBounds.getHeight(),
                  dstFboBounds.getXPos(),
                  dstFboBounds.getYPos(),
                  dstFboBounds.getWidth(),
                  dstFboBounds.getHeight(),
                  filter);
}

// TODO(croot): what if we want to blit a color, depth and/or stencil buffer?
void GLWindow::blitFramebuffer(const Resources::GLFramebufferShPtr& fbo,
                               GLenum attachment,
                               int srcFboX,
                               int srcFboY,
                               int srcFboWidth,
                               int srcFboHeight,
                               int dstFboX,
                               int dstFboY,
                               int dstFboWidth,
                               int dstFboHeight,
                               GLenum filter) {
  // TODO(croot): what renderer do we use here if there are multiple?

  bool isDoubleBuffered = isSettingActive(_settings, IntSetting::USE_DOUBLEBUFFER, true);

  // store the previously set read/draw framebuffers
  GLint currReadFbo;
  GLint currDrawFbo;

  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &currReadFbo));
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &currDrawFbo));

  // set the default framebuffer as the draw buffer
  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
  MAPD_CHECK_GL_ERROR(glDrawBuffer(isDoubleBuffered ? GL_BACK : GL_FRONT));

  if (srcFboWidth < 0 || srcFboHeight < 0) {
    srcFboWidth = fbo->getWidth();
    srcFboHeight = fbo->getHeight();
  }

  if (dstFboWidth < 0 || dstFboHeight < 0) {
    dstFboWidth = getWidth();
    dstFboHeight = getHeight();
  }

  // bind the appropriate attachment from the framebuffer as the read buffer.
  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->getId()));
  MAPD_CHECK_GL_ERROR(glReadBuffer(attachment));

  int srcX0 = srcFboX;
  int srcY0 = srcFboY;
  int srcX1 = srcFboX + srcFboWidth;
  int srcY1 = srcFboY + srcFboHeight;

  int dstX0 = dstFboX;
  int dstY0 = dstFboY;
  int dstX1 = dstFboX + dstFboWidth;
  int dstY1 = dstFboY + dstFboHeight;

  GLbitfield mask = GL_COLOR_BUFFER_BIT;
  if (attachment == GL_DEPTH_ATTACHMENT) {
    mask = GL_DEPTH_BUFFER_BIT;
  } else if (attachment == GL_STENCIL_ATTACHMENT) {
    mask = GL_STENCIL_BUFFER_BIT;
  }

  MAPD_CHECK_GL_ERROR(glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter));

  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, currReadFbo));
  MAPD_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, currDrawFbo));
}

// void GLWindow::makeActiveOnCurrentThread() {
//   RUNTIME_EX_ASSERT(_rendererPtr != nullptr, "GLRenderer is not initialized!");
//   _rendererPtr->makeActiveOnCurrentThread();
// }

// void GLWindow::makeInactive() {
//   RUNTIME_EX_ASSERT(_rendererPtr != nullptr, "GLRenderer is not initialized!");
//   _rendererPtr->makeInactive();
// }

// bool GLWindow::isActiveOnCurrentThread() {
//   RUNTIME_EX_ASSERT(_rendererPtr != nullptr, "GLRenderer is not initialized!");
//   return _rendererPtr->isActiveOnCurrentThread();
// }

// GLRenderer& GLWindow::getRenderer() {
//   RUNTIME_EX_ASSERT(_rendererPtr != nullptr, "GLRenderer is not initialized!");

//   GLRenderer* renderer = dynamic_cast<GLRenderer*>(_rendererPtr.get());
//   CHECK(renderer);

//   return *renderer;
// }

// void GLWindow::_initGLEW() {
//   CHECK(_rendererPtr != nullptr);

//   GLRenderer* renderer = dynamic_cast<GLRenderer*>(_rendererPtr.get());
//   CHECK(renderer);

//   renderer->_initGLEW();
// }

}  // namespace GL

}  // namespace Renderer
