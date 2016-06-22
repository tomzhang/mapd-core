#include "../../QueryRenderer.h"
#include "../../QueryRendererContext.h"
#include <Rendering/Renderer/GL/egl/EglGLRenderer.h>
#include "../QueryFramebuffer.h"
#include "EglQueryRenderCompositorImpl.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/egl/EglDisplayManager.h>

namespace QueryRenderer {
namespace Impl {
namespace EGL {

using ::Rendering::GL::EGL::EglGLRenderer;
using ::Rendering::GL::EGL::EglDisplayShPtr;

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

EglQueryRenderCompositorImpl::EglImage::EglImage(const EglDisplayShPtr& displayPtr, EGLContext ctx, GLuint glId)
    : img(0), displayPtr(displayPtr) {
  PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");

  RUNTIME_EX_ASSERT(eglCreateImageKHR != nullptr, "Necessary KHR_image_base EGL extension is not supported.");

  CHECK(displayPtr);

  img = eglCreateImageKHR(
      displayPtr->getEGLDisplay(), ctx, EGL_GL_TEXTURE_2D, reinterpret_cast<EGLClientBuffer>(glId), NULL);
}

EglQueryRenderCompositorImpl::EglImage::~EglImage() {
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
      _depthEglImgPtr(nullptr),
      _consumedRsrcs() {
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
  // NOTE: all framebuffers that are consumed by the compositor need to be
  // resized before this funcion is called.

  auto glRenderer = getGLRenderer();
  EglGLRenderer* eglRenderer = dynamic_cast<EglGLRenderer*>(glRenderer);
  CHECK(eglRenderer && _framebufferPtr);

  bool doHit = doHitTest();
  bool doDepth = doDepthTest();

  // rebuild the EglImage objects
  _rgbaEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::COLOR_BUFFER)));

  if (doHit) {
    _idEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::ID_BUFFER)));
  }

  if (doDepth) {
    // TODO(croot): fill this out
    THROW_RUNTIME_EX("Depth buffer has yet to be implemented for EGL compositor");
  }

  std::vector<GpuId> gpusMarkedForDeletion;
  int i;
  ::Rendering::GL::Resources::GLTexture2dShPtr texPtr;
  ::Rendering::GL::Resources::GLRenderbufferShPtr rboPtr;

  for (auto& itr : _consumedRsrcs) {
    glRenderer = nullptr;

    std::vector<int> rgbaTexturesMarkedForDeletion;
    auto& rgbaTextures = itr.second.rgbaTextures;
    for (i = 0; i < static_cast<int>(rgbaTextures.size()); ++i) {
      if (rgbaTextures[i].expired()) {
        rgbaTexturesMarkedForDeletion.push_back(i);
      } else if (!glRenderer) {
        glRenderer = rgbaTextures[i].lock()->getGLRenderer();
      }
    }

    for (i = rgbaTexturesMarkedForDeletion.size() - 1; i >= 0; --i) {
      rgbaTextures.erase(rgbaTextures.begin() + i);
    }

    std::vector<int> idTexturesMarkedForDeletion;
    auto& idTextures = itr.second.idTextures;
    for (i = 0; i < static_cast<int>(idTextures.size()); ++i) {
      if (idTextures[i].expired()) {
        idTexturesMarkedForDeletion.push_back(i);
      } else if (!glRenderer) {
        glRenderer = idTextures[i].lock()->getGLRenderer();
      }
    }

    for (i = idTexturesMarkedForDeletion.size() - 1; i >= 0; --i) {
      idTextures.erase(rgbaTextures.begin() + i);
    }

    std::vector<int> rbosMarkedForDeletion;
    auto& rbos = itr.second.rbos;
    for (i = 0; i < static_cast<int>(rbos.size()); ++i) {
      if (rbos[i].expired()) {
        rbosMarkedForDeletion.push_back(i);
      } else if (!glRenderer) {
        glRenderer = rbos[i].lock()->getGLRenderer();
      }
    }

    for (i = rbosMarkedForDeletion.size() - 1; i >= 0; --i) {
      rbos.erase(rbos.begin() + i);
    }

    if (glRenderer) {
      glRenderer->makeActiveOnCurrentThread();

      for (auto& rgbaTex : rgbaTextures) {
        texPtr = rgbaTex.lock();
        CHECK(texPtr);

        glRenderer->bindTexture2d(texPtr);
        MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(texPtr->getTarget(), _rgbaEglImgPtr->img));
      }

      for (auto& idTex : idTextures) {
        texPtr = idTex.lock();
        CHECK(texPtr);

        glRenderer->bindTexture2d(texPtr);
        MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(texPtr->getTarget(), _idEglImgPtr->img));
      }

      // TODO(croot): implement depth buffers
      for (auto& rbo : rbos) {
        rboPtr = rbo.lock();
        CHECK(rboPtr);

        glRenderer->bindRenderbuffer(rboPtr);
        MAPD_CHECK_GL_ERROR(glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, _depthEglImgPtr->img));
      }
    }
  }
}

GLTexture2dShPtr EglQueryRenderCompositorImpl::createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) {
  GpuId gpuId = renderer->getGpuId();
  auto& rsrcStorage = _consumedRsrcs.emplace(gpuId, GLResourceStorage{}).first->second;

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLTexture2dShPtr tex =
      QueryFramebuffer::createFboTexture2d(rsrcMgr, texType, getWidth(), getHeight(), getNumSamples());

  renderer->bindTexture2d(tex);

  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _rgbaEglImgPtr->img));
      rsrcStorage.rgbaTextures.push_back(tex);
      break;
    case FboColorBuffer::ID_BUFFER:
      MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _idEglImgPtr->img));
      rsrcStorage.idTextures.push_back(tex);
      break;
    default:
      CHECK(false);
  }

  return tex;
}

GLRenderbufferShPtr EglQueryRenderCompositorImpl::createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) {
  GpuId gpuId = renderer->getGpuId();
  auto& rsrcStorage = _consumedRsrcs.emplace(gpuId, GLResourceStorage{}).first->second;

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLRenderbufferShPtr rbo =
      QueryFramebuffer::createFboRenderbuffer(rsrcMgr, rboType, getWidth(), getHeight(), getNumSamples());

  renderer->bindRenderbuffer(rbo);
  MAPD_CHECK_GL_ERROR(glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, _depthEglImgPtr->img));

  rsrcStorage.rbos.push_back(rbo);

  return rbo;
}

void EglQueryRenderCompositorImpl::render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) {
  auto ctx = queryRenderer->getContext();
  auto qrmPerGpuDataPtr = ctx->getRootGpuCache()->perGpuData;

  for (auto gpuId : usedGpus) {
    QueryRenderer::renderGpu(gpuId, qrmPerGpuDataPtr, ctx);
  }

  int cnt = 0;
  for (auto itr = usedGpus.begin(); itr != usedGpus.end(); ++itr, ++cnt) {
    if (cnt == 0) {
      QueryRenderer::renderGpu(*itr, qrmPerGpuDataPtr, ctx, 0, 0, 0, 0);
    } else {
      QueryRenderer::renderGpu(*itr, qrmPerGpuDataPtr, ctx);
    }
  }
}

}  // namespace EGL
}  // namespace Impl
}  // namespace QueryRenderer
