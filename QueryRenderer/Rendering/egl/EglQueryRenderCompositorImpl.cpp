#include "../../QueryRenderer.h"
#include <Rendering/Renderer/GL/egl/EglGLRenderer.h>
#include "../QueryFramebuffer.h"
#include "EglQueryRenderCompositorImpl.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/egl/EglDisplayManager.h>
// #include <GL/gl.h>
// #include <GL/glext.h>

namespace QueryRenderer {
namespace Impl {
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

  eglDestroyImageKHR(displayPtr->getEGLDisplay(), img);
}

EglQueryRenderCompositorImpl::EglQueryRenderCompositorImpl(QueryRenderManager* prnt,
                                                           ::Rendering::RendererShPtr& rendererPtr,
                                                           size_t width,
                                                           size_t height,
                                                           size_t numSamples,
                                                           bool doHitTest,
                                                           bool doDepthTest)
    : QueryRenderCompositorImpl(prnt, rendererPtr, width, height, numSamples, doHitTest, doDepthTest),
      _rgbaEglImgPtr(nullptr),
      _idEglImgPtr(nullptr),
      _depthEglImgPtr(nullptr) {
  EglGLRenderer* eglRenderer = dynamic_cast<EglGLRenderer*>(rendererPtr.get());
  CHECK(eglRenderer && _framebufferPtr);

  _rgbaEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::COLOR_BUFFER)));

  if (doHitTest) {
    _idEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::ID_BUFFER)));
  }

  if (doDepthTest) {
    // TODO(croot): fill this out
    THROW_RUNTIME_EX("Depth buffer has yet to be implemented for EGL Compositor");
    //_depthEglImgPtr.reset(new EglImage());
  }
}

EglQueryRenderCompositorImpl::~EglQueryRenderCompositorImpl() {
}

void EglQueryRenderCompositorImpl::_resizeImpl(size_t width, size_t height) {
  // TODO(croot): do we need to do anything here? I think the resize of the parent
  // class's framebuffer will handle the resize appropriately for us.
  // THROW_RUNTIME_EX("EglQueryRenderCompositorImpl::_resizeImpl() not implemented yet");
}

GLTexture2dShPtr EglQueryRenderCompositorImpl::createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
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

GLRenderbufferShPtr EglQueryRenderCompositorImpl::createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) {
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLRenderbufferShPtr rbo = QueryFramebuffer::createFboRenderbuffer(rsrcMgr, rboType, getWidth(), getHeight());

  renderer->bindRenderbuffer(rbo);
  MAPD_CHECK_GL_ERROR(glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, _depthEglImgPtr->img));

  return rbo;
}

void EglQueryRenderCompositorImpl::render(QueryRenderer* queryRenderer) {
  QueryRendererContext* ctx = queryRenderer->getContext();
  QueryRenderer::PerGpuDataMap* perGpuData = queryRenderer->getPerGpuData();

  auto itr = perGpuData->begin();
  int cnt = 0;
  for (; itr != perGpuData->end(); ++itr, ++cnt) {
    if (cnt == 0) {
      QueryRenderer::renderGpu(itr->first, perGpuData, ctx, 0, 0, 0, 0);
    } else {
      QueryRenderer::renderGpu(itr->first, perGpuData, ctx, 0, 0, 0, 1);
    }
  }
}

}  // namespace EGL
}  // namespace Impl
}  // namespace QueryRenderer
