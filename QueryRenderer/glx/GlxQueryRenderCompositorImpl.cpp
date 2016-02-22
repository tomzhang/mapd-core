#include "../QueryRenderer.h"
#include <Rendering/Renderer/GL/glx/GlxGLRenderer.h>
#include "../QueryFramebuffer.h"
#include "GlxQueryRenderCompositorImpl.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/glx/X11DisplayManager.h>
#include "../PngData.h"
// #include <GL/gl.h>
// #include <GL/glext.h>

namespace QueryRenderer {
namespace Impl {
namespace GLX {

using ::Rendering::GL::GLX::GlxGLRenderer;
using ::Rendering::GL::GLX::X11DisplayShPtr;

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLRenderbufferShPtr;

GlxQueryRenderCompositorImpl::GlxQueryRenderCompositorImpl(QueryRenderer* prnt,
                                                           ::Rendering::GL::GLRenderer* renderer,
                                                           size_t width,
                                                           size_t height,
                                                           size_t numSamples,
                                                           bool doHitTest,
                                                           bool doDepthTest)
    : QueryRenderCompositorImpl(prnt, renderer, width, height, numSamples, doHitTest, doDepthTest),
      _rgbaTextures(),
      _idTextures(),
      _rbos() {
  _renderer = dynamic_cast<GlxGLRenderer*>(renderer);
  CHECK(_renderer != nullptr);

  RUNTIME_EX_ASSERT(
      glXCopyImageSubDataNV,
      "NV_copy_image GLX extension is not supported. Cannot initialize compositor for multi-gpu rendering.");
}

GlxQueryRenderCompositorImpl::~GlxQueryRenderCompositorImpl() {
}

void GlxQueryRenderCompositorImpl::_resizeImpl(size_t width, size_t height) {
  for (auto& rgbaItr : _rgbaTextures) {
    rgbaItr.second->resize(width, height);
  }

  for (auto& idItr : _idTextures) {
    idItr.second->resize(width, height);
  }

  for (auto& rboItr : _rbos) {
    rboItr.second->resize(width, height);
  }
}

GLTexture2dShPtr GlxQueryRenderCompositorImpl::createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) {
  int width = getWidth();
  int height = getHeight();

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLTexture2dShPtr tex = QueryFramebuffer::createFboTexture2d(rsrcMgr, texType, width, height);

  _renderer->makeActiveOnCurrentThread();
  GLResourceManagerShPtr myRsrcMgr = _renderer->getResourceManager();
  GLTexture2dShPtr myTex = QueryFramebuffer::createFboTexture2d(myRsrcMgr, texType, width, height);

  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      _rgbaTextures.insert({tex, myTex});
      break;
    case FboColorBuffer::ID_BUFFER:
      _idTextures.insert({tex, myTex});
      break;
    default:
      CHECK(false);
  }

  // reset the active renderer
  renderer->makeActiveOnCurrentThread();

  return tex;
}

GLRenderbufferShPtr GlxQueryRenderCompositorImpl::createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) {
  int width = getWidth();
  int height = getHeight();

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLRenderbufferShPtr rbo = QueryFramebuffer::createFboRenderbuffer(rsrcMgr, rboType, getWidth(), getHeight());

  _renderer->makeActiveOnCurrentThread();
  GLResourceManagerShPtr myRsrcMgr = _renderer->getResourceManager();
  GLRenderbufferShPtr myRbo = QueryFramebuffer::createFboRenderbuffer(myRsrcMgr, rboType, width, height);

  _rbos.insert({rbo, myRbo});

  // reset the active renderer
  renderer->makeActiveOnCurrentThread();

  return rbo;
}

void GlxQueryRenderCompositorImpl::deleteFboTexture2d(
    const ::Rendering::GL::Resources::GLTexture2dShPtr& texture2dPtr) {
  THROW_RUNTIME_EX("deleteFboTexture2d() method is not yet implemented");
}

void GlxQueryRenderCompositorImpl::deleteFboRenderbuffer(
    const ::Rendering::GL::Resources::GLRenderbufferShPtr& renderbufferPtr) {
  THROW_RUNTIME_EX("deleteFboRenderbuffer() method is not yet implemented");
}

void GlxQueryRenderCompositorImpl::render(QueryRenderer* queryRenderer) {
  QueryRendererContext* ctx = queryRenderer->getContext();
  QueryRenderer::PerGpuDataMap* perGpuData = queryRenderer->getPerGpuData();

  GLXContext myGlxCtx = _renderer->getGLXContext();
  X11DisplayShPtr myDisplayPtr = _renderer->getXDisplayPtr();

  bool doHitTest = ctx->doHitTest();
  bool doDepthTest = ctx->doDepthTest();

  size_t width = ctx->getWidth();
  size_t height = ctx->getHeight();

  GLTexture2dShPtr rgbaTex, idTex;
  GLRenderbufferShPtr depthRbo;

  auto itr = perGpuData->begin();
  int cnt = 0;
  for (; itr != perGpuData->end(); ++itr, ++cnt) {
    GlxGLRenderer* glxRenderer = dynamic_cast<GlxGLRenderer*>(itr->second.qrmGpuData->rendererPtr.get());
    CHECK(glxRenderer != nullptr);

    GLXContext glxCtx = glxRenderer->getGLXContext();
    X11DisplayShPtr displayPtr = glxRenderer->getXDisplayPtr();

    CHECK(displayPtr == myDisplayPtr);

    if (cnt == 0) {
      QueryRenderer::renderGpu(itr->first, perGpuData, ctx, 0, 0, 0, 0);
    } else {
      QueryRenderer::renderGpu(itr->first, perGpuData, ctx, 0, 0, 0, 0);
    }

    // TODO(croot): do we need to do a flush/finish before copying?
    // or will the copy take care of that for us?

    rgbaTex = itr->second.framebufferPtr->getColorTexture2d(FboColorBuffer::COLOR_BUFFER);
    auto rgbaItr = _rgbaTextures.find(rgbaTex);
    CHECK(rgbaItr != _rgbaTextures.end());

    glXCopyImageSubDataNV(displayPtr.get(),
                          glxCtx,
                          rgbaTex->getId(),
                          rgbaTex->getTarget(),
                          0,
                          0,
                          0,
                          0,
                          myGlxCtx,
                          rgbaItr->second->getId(),
                          rgbaItr->second->getTarget(),
                          0,
                          0,
                          0,
                          0,
                          width,
                          height,
                          0);

    _renderer->makeActiveOnCurrentThread();
    _renderer->bindTexture2d(rgbaItr->second);

    std::shared_ptr<unsigned char> pixels(new unsigned char[width * height * 4],
                                          std::default_delete<unsigned char[]>());
    unsigned char* rawPixels = pixels.get();
    _renderer->getBoundTexture2dPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, rawPixels);

    PngData pngData(width, height, pixels);
    pngData.writeToFile("color_" + std::to_string(itr->first) + ".png");

    if (doHitTest) {
      idTex = itr->second.framebufferPtr->getColorTexture2d(FboColorBuffer::ID_BUFFER);
    }

    if (doDepthTest) {
      depthRbo = itr->second.framebufferPtr->getRenderbuffer(FboRenderBuffer::DEPTH_BUFFER);
    }
    // glXCopyImageSubDataNV,
  }
}

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer
