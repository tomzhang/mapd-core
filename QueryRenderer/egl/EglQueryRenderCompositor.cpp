#include <Rendering/Renderer/GL/egl/EglGLRenderer.h>
#include "../QueryFramebuffer.h"
#include "EglQueryRenderCompositor.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/egl/EglDisplayManager.h>
// #include <GL/gl.h>
// #include <GL/glext.h>

namespace QueryRenderer {

namespace EGL {

using ::Rendering::GL::EGL::EglGLRenderer;
using ::Rendering::GL::EGL::EglDisplayShPtr;

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

EglImage::EglImage(const EglDisplayShPtr& displayPtr, EGLContext ctx, GLuint glId) : img(0), displayPtr(displayPtr) {
  PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");

  RUNTIME_EX_ASSERT(eglCreateImageKHR != nullptr, "Necessary KHR_image_base EGL extension is not supported.");

  CHECK(displayPtr);

  img = eglCreateImageKHR(
      displayPtr->getEGLDisplay(), ctx, EGL_GL_TEXTURE_2D, reinterpret_cast<EGLClientBuffer>(glId), NULL);
}

EglImage::~EglImage() {
  PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");

  CHECK(eglDestroyImageKHR);

  eglDestroyImage(displayPtr->getEGLDisplay(), img);
}

EglQueryRenderCompositor::EglQueryRenderCompositor(::Rendering::GL::GLRenderer* renderer,
                                                   size_t width,
                                                   size_t height,
                                                   size_t numSamples,
                                                   bool doHitTest,
                                                   bool doDepthTest)
    : QueryRenderCompositor(renderer, width, height, numSamples, doHitTest, doDepthTest),
      _rgbaEglImgPtr(nullptr),
      _idEglImgPtr(nullptr) {
  // _depthEglImgPtr(nullptr) {

  EglGLRenderer* eglRenderer = dynamic_cast<EglGLRenderer*>(renderer);
  CHECK(eglRenderer);

  CHECK(_framebufferPtr);

  _rgbaEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::COLOR_BUFFER)));

  if (doHitTest) {
    _idEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::ID_BUFFER)));
  }
}

EglQueryRenderCompositor::~EglQueryRenderCompositor() {
}

void EglQueryRenderCompositor::resize(size_t width, size_t height) {
}

GLTexture2dShPtr EglQueryRenderCompositor::_createFboTexture2dImpl(::Rendering::GL::GLRenderer* renderer,
                                                                   FboColorBuffer texType) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLTexture2dShPtr tex = QueryFramebuffer::createFboTexture2d(rsrcMgr, texType, getWidth(), getHeight());

  renderer->bindTexture2d(tex);

  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _rgbaEglImgPtr->img));
      break;
    case FboColorBuffer::ID_BUFFER:
      MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _idEglImgPtr->img));
      break;
    default:
      CHECK(false);
  }

  return tex;
}

GLRenderbufferShPtr EglQueryRenderCompositor::_createFboRenderbufferImpl(::Rendering::GL::GLRenderer* renderer,
                                                                         FboRenderBuffer rboType) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLRenderbufferShPtr rbo = QueryFramebuffer::createFboRenderbuffer(rsrcMgr, rboType, getWidth(), getHeight());

  renderer->bindRenderbuffer(rbo);
  MAPD_CHECK_GL_ERROR(glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, _depthEglImgPtr->img));

  return rbo;
}

}  // namespace EGL

}  // namespace QueryRenderer
