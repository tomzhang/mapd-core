#include "../../QueryRenderManager.h"
#include "../../QueryRenderer.h"
#include "../../QueryRendererContext.h"
#include "../../Scales/Scale.h"
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
                                                           const size_t width,
                                                           const size_t height,
                                                           const size_t numSamples,
                                                           const bool doHitTest,
                                                           const bool doDepthTest,
                                                           const bool supportsInt64)
    : QueryRenderCompositorImpl(prnt, rendererPtr, width, height, numSamples, doHitTest, doDepthTest) {
  EglGLRenderer* eglRenderer = dynamic_cast<EglGLRenderer*>(rendererPtr.get());
  CHECK(eglRenderer && _framebufferPtr);

  _rgbaEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::COLOR_BUFFER)));

  if (doHitTest) {
    _id1AEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                      eglRenderer->getEGLContext(),
                                      _framebufferPtr->getId(FboColorBuffer::ID1A_BUFFER)));

    if (supportsInt64) {
      _id1BEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                        eglRenderer->getEGLContext(),
                                        _framebufferPtr->getId(FboColorBuffer::ID1B_BUFFER)));
    }

    _id2EglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                     eglRenderer->getEGLContext(),
                                     _framebufferPtr->getId(FboColorBuffer::ID2_BUFFER)));
  }

  if (doDepthTest) {
    // TODO(croot): fill this out
    THROW_RUNTIME_EX("Depth buffer has yet to be implemented for EGL Compositor");
    //_depthEglImgPtr.reset(new EglImage());
  }
}

EglQueryRenderCompositorImpl::~EglQueryRenderCompositorImpl() {}

void EglQueryRenderCompositorImpl::_resizeImpl(size_t width, size_t height) {
  // NOTE: all framebuffers that are consumed by the compositor need to be
  // resized before this funcion is called.

  auto glRenderer = getGLRenderer();
  EglGLRenderer* eglRenderer = dynamic_cast<EglGLRenderer*>(glRenderer);
  CHECK(eglRenderer && _framebufferPtr);

  bool doHit = doHitTest();
  bool doDepth = doDepthTest();
  bool useInt64 = (_id1BEglImgPtr != nullptr);

  // rebuild the EglImage objects
  _rgbaEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                    eglRenderer->getEGLContext(),
                                    _framebufferPtr->getId(FboColorBuffer::COLOR_BUFFER)));

  if (doHit) {
    _id1AEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                      eglRenderer->getEGLContext(),
                                      _framebufferPtr->getId(FboColorBuffer::ID1A_BUFFER)));

    if (useInt64) {
      _id1BEglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                        eglRenderer->getEGLContext(),
                                        _framebufferPtr->getId(FboColorBuffer::ID1B_BUFFER)));
    }

    _id2EglImgPtr.reset(new EglImage(eglRenderer->getEGLDisplayPtr(),
                                     eglRenderer->getEGLContext(),
                                     _framebufferPtr->getId(FboColorBuffer::ID2_BUFFER)));
  }

  if (doDepth) {
    // TODO(croot): fill this out
    THROW_RUNTIME_EX("Depth buffer has yet to be implemented for EGL compositor");
  }

  int i;

  // resize the accum textures and rebuild the EglImage objects for them
  CHECK(_accumTexPtrArrayMap.size() == _accumEglImgPtrArrayMap.size());
  for (i = 0; i < static_cast<int>(_accumTexPtrArrayMap.size()); ++i) {
    _accumTexPtrArrayMap[i]->resize(width, height);
    _accumEglImgPtrArrayMap[i].reset(
        new EglImage(eglRenderer->getEGLDisplayPtr(), eglRenderer->getEGLContext(), _accumTexPtrArrayMap[i]->getId()));
  }

  ::Rendering::GL::Resources::GLTexture2dShPtr texPtr;
  ::Rendering::GL::Resources::GLRenderbufferShPtr rboPtr;

  for (auto& itr : _consumedRsrcs) {
    glRenderer = nullptr;

    auto& rgbaTextures = itr.second.rgbaTextures;
    rgbaTextures.erase(
        std::remove_if(
            rgbaTextures.begin(), rgbaTextures.end(), [](auto& rgbaTexture) { return rgbaTexture.expired(); }),
        rgbaTextures.end());

    if (rgbaTextures.size()) {
      glRenderer = rgbaTextures[0].lock()->getGLRenderer();
    }

    auto& id1ATextures = itr.second.id1ATextures;
    id1ATextures.erase(
        std::remove_if(
            id1ATextures.begin(), id1ATextures.end(), [](auto& id1ATexture) { return id1ATexture.expired(); }),
        id1ATextures.end());

    if (!glRenderer && id1ATextures.size()) {
      glRenderer = id1ATextures[0].lock()->getGLRenderer();
    }

    auto& id1BTextures = itr.second.id1BTextures;
    if (useInt64) {
      id1BTextures.erase(
          std::remove_if(
              id1BTextures.begin(), id1BTextures.end(), [](auto& id1BTexture) { return id1BTexture.expired(); }),
          id1BTextures.end());

      CHECK(id1ATextures.size() == id1BTextures.size());
    }

    auto& id2Textures = itr.second.id2Textures;
    id2Textures.erase(
        std::remove_if(id2Textures.begin(), id2Textures.end(), [](auto& id2Texture) { return id2Texture.expired(); }),
        id2Textures.end());

    CHECK(id1ATextures.size() == id2Textures.size());

    auto& rbos = itr.second.rbos;
    rbos.erase(std::remove_if(rbos.begin(), rbos.end(), [](auto& rbo) { return rbo.expired(); }), rbos.end());

    if (!glRenderer && rbos.size()) {
      glRenderer = rbos[0].lock()->getGLRenderer();
    }

    if (!glRenderer) {
      for (auto& map : itr.second.registeredAccumTxts) {
        for (auto mapItr : map) {
          auto shptr = mapItr.second.lock();
          if (shptr) {
            glRenderer = shptr->getGLRenderer();
            break;
          }
        }
        if (glRenderer) {
          break;
        }
      }
    }

    if (glRenderer) {
      glRenderer->makeActiveOnCurrentThread();

      for (auto& rgbaTex : rgbaTextures) {
        texPtr = rgbaTex.lock();
        CHECK(texPtr);

        glRenderer->bindTexture2d(texPtr);
        MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(texPtr->getTarget(), _rgbaEglImgPtr->img));
      }

      for (auto& id1ATex : id1ATextures) {
        texPtr = id1ATex.lock();
        CHECK(texPtr);

        glRenderer->bindTexture2d(texPtr);
        MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(texPtr->getTarget(), _id1AEglImgPtr->img));
      }

      if (useInt64) {
        for (auto& id1BTex : id1BTextures) {
          texPtr = id1BTex.lock();
          CHECK(texPtr);

          glRenderer->bindTexture2d(texPtr);
          MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(texPtr->getTarget(), _id1BEglImgPtr->img));
        }
      }

      for (auto& id2Tex : id2Textures) {
        texPtr = id2Tex.lock();
        CHECK(texPtr);

        glRenderer->bindTexture2d(texPtr);
        MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(texPtr->getTarget(), _id2EglImgPtr->img));
      }

      // TODO(croot): implement depth buffers
      for (auto& rbo : rbos) {
        rboPtr = rbo.lock();
        CHECK(rboPtr);

        glRenderer->bindRenderbuffer(rboPtr);
        MAPD_CHECK_GL_ERROR(glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, _depthEglImgPtr->img));
      }

      CHECK(itr.second.registeredAccumTxts.size() <= _accumTexPtrArrayMap.size());

      for (i = 0; i < static_cast<int>(itr.second.registeredAccumTxts.size()); ++i) {
        auto& map = itr.second.registeredAccumTxts[i];
        for (auto mapItr : map) {
          auto shptr = mapItr.second.lock();
          CHECK(shptr);
          glRenderer->bindTexture2d(shptr);
          MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(shptr->getTarget(), _accumEglImgPtrArrayMap[i]->img));
        }
      }
    }
  }

  if (glRenderer != eglRenderer) {
    eglRenderer->makeActiveOnCurrentThread();
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
    case FboColorBuffer::ID1A_BUFFER:
      MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _id1AEglImgPtr->img));
      rsrcStorage.id1ATextures.push_back(tex);
      break;
    case FboColorBuffer::ID1B_BUFFER:
      MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _id1BEglImgPtr->img));
      rsrcStorage.id1BTextures.push_back(tex);
      break;
    case FboColorBuffer::ID2_BUFFER:
      MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _id2EglImgPtr->img));
      rsrcStorage.id2Textures.push_back(tex);
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

void EglQueryRenderCompositorImpl::_initAccumResources(size_t width, size_t height, size_t depth) {
  GLRenderer* currRenderer;
  bool resetRenderer = false;

  if (depth > _accumTexPtrArrayMap.size()) {
    CHECK(_framebufferPtr);
    auto myRenderer = _framebufferPtr->getGLRenderer();
    CHECK(myRenderer);

    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != myRenderer))) {
      myRenderer->makeActiveOnCurrentThread();
    }

    auto myRsrcMgr = myRenderer->getResourceManager();

    auto eglRenderer = dynamic_cast<EglGLRenderer*>(myRenderer);
    CHECK(eglRenderer);
    auto eglDisplayPtr = eglRenderer->getEGLDisplayPtr();
    auto eglContextPtr = eglRenderer->getEGLContext();

    auto diff = depth - _accumTexPtrArrayMap.size();
    for (size_t i = 0; i < diff; ++i) {
      _accumTexPtrArrayMap.push_back(
          myRsrcMgr->createTexture2d(width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT));
      _accumEglImgPtrArrayMap.emplace_back(
          new EglImage(eglDisplayPtr, eglContextPtr, _accumTexPtrArrayMap.back()->getId()));
    }
  }

  CHECK(_accumTexPtrArrayMap.size() == _accumEglImgPtrArrayMap.size());

  if (resetRenderer) {
    currRenderer->makeActiveOnCurrentThread();
  }
}

void EglQueryRenderCompositorImpl::_cleanupAccumResources() {
  GLRenderer* currRenderer;
  bool resetRenderer = false;

  CHECK(_accumTexPtrArrayMap.size() == _accumEglImgPtrArrayMap.size());

  // TODO(croot): should we cleanup any expired weak ptrs?
  int maxSize = -1;
  size_t sz;
  for (auto gpuItr : _consumedRsrcs) {
    for (int i = gpuItr.second.registeredAccumTxts.size() - 1; i > maxSize; --i) {
      sz = gpuItr.second.registeredAccumTxts[i].size();
      if (sz > 0) {
        maxSize = i;
        break;
      }
    }
  }

  maxSize++;

  if (maxSize == 0) {
    CHECK(_framebufferPtr);
    auto myRenderer = _framebufferPtr->getGLRenderer();
    CHECK(myRenderer);

    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != myRenderer))) {
      myRenderer->makeActiveOnCurrentThread();
    }

    _accumTexPtrArrayMap.clear();
    _accumEglImgPtrArrayMap.clear();
  } else if (maxSize < static_cast<int>(_accumTexPtrArrayMap.size())) {
    CHECK(_framebufferPtr);
    auto myRenderer = _framebufferPtr->getGLRenderer();
    CHECK(myRenderer);

    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != myRenderer))) {
      myRenderer->makeActiveOnCurrentThread();
    }

    _accumTexPtrArrayMap.resize(maxSize);
    _accumEglImgPtrArrayMap.resize(maxSize);

    for (auto gpuItr : _consumedRsrcs) {
      gpuItr.second.registeredAccumTxts.resize(maxSize);
    }
  }

  if (resetRenderer) {
    currRenderer->makeActiveOnCurrentThread();
  }
}

void EglQueryRenderCompositorImpl::registerAccumulatorTexture(::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                                              size_t accumIdx,
                                                              size_t numTexturesInArray) {
  // TODO(croot): make thread safe?

  CHECK(accumIdx < numTexturesInArray);

  _initAccumResources(getWidth(), getHeight(), numTexturesInArray);

  auto renderer = tex->getGLRenderer();
  auto gpuId = renderer->getGpuId();
  auto& rsrcStorage = _consumedRsrcs.emplace(gpuId, GLResourceStorage{}).first->second;

  if (numTexturesInArray > rsrcStorage.registeredAccumTxts.size()) {
    rsrcStorage.registeredAccumTxts.resize(numTexturesInArray);
  }

  // TODO(croot): should we be careful and check to see if the tex has already been added?
  renderer->bindTexture2d(tex);
  MAPD_CHECK_GL_ERROR(glEGLImageTargetTexture2DOES(tex->getTarget(), _accumEglImgPtrArrayMap[accumIdx]->img));

  rsrcStorage.registeredAccumTxts[accumIdx].emplace(tex.get(), tex);
}

void EglQueryRenderCompositorImpl::unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                                                size_t accumIdx) {
  auto gpuId = tex->getGLRenderer()->getGpuId();
  auto gpuItr = _consumedRsrcs.find(gpuId);
  CHECK(gpuItr != _consumedRsrcs.end());

  // TODO(croot): if the tex is not found at the accumIdx, should we indicate where it
  // can be found?

  // TODO(croot): make thread safe?

  CHECK(accumIdx < gpuItr->second.registeredAccumTxts.size());
  auto& map = gpuItr->second.registeredAccumTxts[accumIdx];

  auto itr = map.find(tex.get());
  CHECK(itr != map.end());

  map.erase(itr);

  _cleanupAccumResources();
}

void EglQueryRenderCompositorImpl::_postPassPerGpuCB(::Rendering::GL::GLRenderer* renderer,
                                                     QueryFramebufferUqPtr& framebufferPtr,
                                                     size_t width,
                                                     size_t height,
                                                     bool doHitTest,
                                                     bool doDepthTest,
                                                     int passCnt,
                                                     ScaleShPtr& accumulatorScalePtr,
                                                     int accumulatorCnt) {}

void EglQueryRenderCompositorImpl::_compositePass(const std::set<GpuId>& usedGpus,
                                                  size_t width,
                                                  size_t height,
                                                  bool doHitTest,
                                                  bool doDepthTest,
                                                  int passCnt,
                                                  ScaleShPtr& accumulatorScalePtr) {
  if (accumulatorScalePtr) {
    CHECK(_framebufferPtr);
    auto myRenderer = _framebufferPtr->getGLRenderer();
    CHECK(myRenderer);
    myRenderer->makeActiveOnCurrentThread();

    accumulatorScalePtr->renderAccumulation(myRenderer, myRenderer->getGpuId());
  }
}

void EglQueryRenderCompositorImpl::render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) {
  auto ctx = queryRenderer->getContext();
  auto qrmPerGpuDataPtr = ctx->getRootGpuCache()->perGpuData;

  QueryRenderer::renderPasses(qrmPerGpuDataPtr,
                              ctx,
                              usedGpus,
                              false,
                              std::bind(&EglQueryRenderCompositorImpl::_postPassPerGpuCB,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6,
                                        std::placeholders::_7,
                                        std::placeholders::_8,
                                        std::placeholders::_9),
                              std::bind(&EglQueryRenderCompositorImpl::_compositePass,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6,
                                        std::placeholders::_7));
}

}  // namespace EGL
}  // namespace Impl
}  // namespace QueryRenderer
