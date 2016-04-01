#include "../../PngData.h"
#include "../../QueryRendererContext.h"
#include <Rendering/Renderer/GL/glx/GlxGLRenderer.h>
#include "../QueryFramebuffer.h"
#include "GlxQueryRenderCompositorImpl.h"
#include "shaders/compositor_vert.h"
#include "shaders/compositor_frag.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/Resources/GLBufferLayout.h>
#include <Rendering/Renderer/GL/Resources/GLTexture2d.h>
#include <Rendering/Renderer/GL/glx/X11DisplayManager.h>

namespace QueryRenderer {
namespace Impl {
namespace GLX {

using ::Rendering::GL::GLX::GlxGLRenderer;
using ::Rendering::GL::GLX::X11DisplayShPtr;

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;

using ::Rendering::GL::Resources::GLRenderbufferShPtr;
using ::Rendering::GL::Resources::GLInterleavedBufferLayout;
using ::Rendering::GL::Resources::GLInterleavedBufferLayoutShPtr;
using ::Rendering::GL::Resources::GLShaderShPtr;
using ::Rendering::GL::Resources::GLTexture2dShPtr;
using ::Rendering::GL::Resources::GLTexture2dArrayShPtr;
using ::Rendering::GL::Resources::GLVertexBufferShPtr;
using ::Rendering::GL::Resources::GLVertexArrayShPtr;

using ::Rendering::GL::Resources::GLTexture2dSampleProps;

#ifdef GLEW_MX
#undef glxewGetContext
#define glxewGetContext _renderer->glxewGetContext
#endif

GlxQueryRenderCompositorImpl::GlxQueryRenderCompositorImpl(QueryRenderManager* prnt,
                                                           ::Rendering::RendererShPtr& rendererPtr,
                                                           size_t width,
                                                           size_t height,
                                                           size_t numSamples,
                                                           bool doHitTest,
                                                           bool doDepthTest)
    : QueryRenderCompositorImpl(prnt, rendererPtr, width, height, numSamples, doHitTest, doDepthTest),
      _rendererPtr(rendererPtr),
      _renderer(nullptr),
      _rectvbo(nullptr),
      _shader(nullptr),
      _vao(nullptr),
      _rgbaTextureArray(nullptr),
      _idTextureArray(nullptr),
      _rgbaTextures(),
      _idTextures(),
      _rbos() {
  CHECK(rendererPtr);
  _renderer = dynamic_cast<GlxGLRenderer*>(rendererPtr.get());
  CHECK(_renderer != nullptr);

  RUNTIME_EX_ASSERT(
      GLXEW_NV_copy_image,
      "NV_copy_image GLX extension is not supported. Cannot initialize compositor for multi-gpu rendering.");

  _initResources(prnt);
}

void GlxQueryRenderCompositorImpl::_initResources(QueryRenderManager* queryRenderManager) {
  _renderer->makeActiveOnCurrentThread();
  GLResourceManagerShPtr rsrcMgr = _renderer->getResourceManager();

  // create a rectangle vertex buffer that will cover the entire buffer
  // and textured with the textures from all the gpus
  GLInterleavedBufferLayoutShPtr bufferLayout(new GLInterleavedBufferLayout());
  bufferLayout->addAttribute<float, 2>("pos");
  bufferLayout->addAttribute<float, 2>("texCoords");
  _rectvbo = rsrcMgr->createVertexBuffer<float>(
      {-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0}, bufferLayout);

  _shader = rsrcMgr->createShader(Compositor_vert::source, Compositor_frag::source);
  _renderer->bindShader(_shader);

  // TODO(croot): automate the texture image unit binding
  _shader->setSamplerTextureImageUnit("rgbaArraySampler", GL_TEXTURE0);
  _shader->setSamplerTextureImageUnit("idArraySampler", GL_TEXTURE1);

  _vao = rsrcMgr->createVertexArray({{_rectvbo, {}}});

  size_t width = getWidth();
  size_t height = getHeight();
  size_t depth = queryRenderManager->getPerGpuData()->size();
  if (depth == 0) {
    // the texture array needs to be initialized to something
    depth = 1;
  }
  GLTexture2dSampleProps sampleProps;
  size_t numSamples = getNumSamples();

  _rgbaTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_RGBA8, sampleProps, numSamples);

  if (doHitTest()) {
    _idTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI, sampleProps, numSamples);
  }
}

GlxQueryRenderCompositorImpl::~GlxQueryRenderCompositorImpl() {
  // need to make the renderer active to properly delete the GL resources
  // TODO(croot): reset to previously active renderer?
  if (!_rendererPtr.expired()) {
    _renderer->makeActiveOnCurrentThread();
  }
}

void GlxQueryRenderCompositorImpl::_resizeImpl(size_t width, size_t height) {
  if (_rgbaTextureArray) {
    _rgbaTextureArray->resize(width, height);
  }

  if (_idTextureArray) {
    CHECK(doHitTest());
    _idTextureArray->resize(width, height);
  }
}

GLTexture2dShPtr GlxQueryRenderCompositorImpl::createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) {
  size_t width = getWidth();
  size_t height = getHeight();

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLTexture2dShPtr tex = QueryFramebuffer::createFboTexture2d(rsrcMgr, texType, width, height);

  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      _rgbaTextures.insert(tex.get());
      break;
    case FboColorBuffer::ID_BUFFER:
      _idTextures.insert(tex.get());
      break;
    default:
      CHECK(false);
  }

  return tex;
}

GLRenderbufferShPtr GlxQueryRenderCompositorImpl::createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) {
  size_t width = getWidth();
  size_t height = getHeight();

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLRenderbufferShPtr rbo = QueryFramebuffer::createFboRenderbuffer(rsrcMgr, rboType, width, height);

  _rbos.insert(rbo.get());

  return rbo;
}

void GlxQueryRenderCompositorImpl::addFboTexture2d(::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                                   FboColorBuffer texType) {
  size_t width = getWidth();
  size_t height = getHeight();

  RUNTIME_EX_ASSERT(width == tex->getWidth() && height == tex->getHeight(),
                    "Cannot add texture 2d object to compositor - invalid dimensions: tex " +
                        std::to_string(tex->getWidth()) + "x" + std::to_string(tex->getHeight()) + " != compositor " +
                        std::to_string(width) + "x" + std::to_string(height) + ".");

  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      CHECK(_rgbaTextures.find(tex.get()) == _rgbaTextures.end());
      _rgbaTextures.insert(tex.get());
      break;
    case FboColorBuffer::ID_BUFFER:
      CHECK(_idTextures.find(tex.get()) == _idTextures.end());
      _idTextures.insert(tex.get());
      break;
    default:
      CHECK(false);
  }
}

void GlxQueryRenderCompositorImpl::addFboRenderbuffer(::Rendering::GL::Resources::GLRenderbufferShPtr& rbo,
                                                      FboRenderBuffer rboType) {
  size_t width = getWidth();
  size_t height = getHeight();

  RUNTIME_EX_ASSERT(width == rbo->getWidth() && height == rbo->getHeight(),
                    "Cannot add renderbuffer object to compositor - invalid dimensions: rbo " +
                        std::to_string(rbo->getWidth()) + "x" + std::to_string(rbo->getHeight()) + " != compositor " +
                        std::to_string(width) + "x" + std::to_string(height) + ".");

  CHECK(_rbos.find(rbo.get()) == _rbos.end());
  _rbos.insert(rbo.get());
}

void GlxQueryRenderCompositorImpl::deleteFboTexture2d(::Rendering::GL::Resources::GLTexture2d* texture2dPtr) {
  // TODO(croot): make thread safe?
  auto rgbaItr = _rgbaTextures.find(texture2dPtr);
  if (rgbaItr == _rgbaTextures.end()) {
    auto idItr = _idTextures.find(texture2dPtr);
    if (idItr != _idTextures.end()) {
      _idTextures.erase(idItr);
    }

    CHECK(_idTextureArray);
    _idTextureArray->resize(_idTextures.size());
  } else {
    _rgbaTextures.erase(rgbaItr);

    CHECK(_rgbaTextureArray);
    _rgbaTextureArray->resize(_rgbaTextures.size());
  }
}

void GlxQueryRenderCompositorImpl::deleteFboRenderbuffer(::Rendering::GL::Resources::GLRenderbuffer* renderbufferPtr) {
  // TODO(croot): make thread safe?
  auto rboItr = _rbos.find(renderbufferPtr);
  if (rboItr != _rbos.end()) {
    _rbos.erase(rboItr);

    // TODO(croot): need to resize the depth texture array, if/when
    // that gets implemented.
  }
}

void GlxQueryRenderCompositorImpl::render(QueryRenderer* queryRenderer) {
  QueryRendererContext* ctx = queryRenderer->getContext();
  QueryRenderer::PerGpuDataMap* perGpuData = queryRenderer->getPerGpuData();

  GLXContext myGlxCtx = _renderer->getGLXContext();
  X11DisplayShPtr myDisplayPtr = _renderer->getXDisplayPtr();

  bool doHitTest = ctx->doHitTest();
  // bool doDepthTest = ctx->doDepthTest();

  size_t width = ctx->getWidth();
  size_t height = ctx->getHeight();

  GLTexture2dShPtr rgbaTex, idTex;
  GLRenderbufferShPtr depthRbo;

  CHECK(_rgbaTextureArray && _rgbaTextures.size());
  if (_rgbaTextureArray->getDepth() != _rgbaTextures.size()) {
    _renderer->makeActiveOnCurrentThread();
    _rgbaTextureArray->resize(_rgbaTextures.size());
    CHECK(_rgbaTextures.size() == _rgbaTextureArray->getDepth());
  }

  if (doHitTest) {
    CHECK(_idTextureArray && _idTextures.size() && _idTextures.size() == _rgbaTextures.size());
    if (_idTextureArray->getDepth() != _idTextures.size()) {
      _renderer->makeActiveOnCurrentThread();
      _idTextureArray->resize(_idTextures.size());
      CHECK(_idTextures.size() == _idTextureArray->getDepth());
    }
  }
  // if (doDepthTest) {
  //   CHECK(_rbos.size() == _)
  // }

  auto itr = perGpuData->begin();
  int cnt = 0;
  for (; itr != perGpuData->end(); ++itr, ++cnt) {
    QueryRenderer::renderGpu(itr->first, perGpuData, ctx, 0, 0, 0, 0);
  }

  itr = perGpuData->begin();
  cnt = 0;
  for (; itr != perGpuData->end(); ++itr, ++cnt) {
    GlxGLRenderer* glxRenderer = dynamic_cast<GlxGLRenderer*>(itr->second.getQRMGpuData()->rendererPtr.get());
    CHECK(glxRenderer != nullptr);

    GLXContext glxCtx = glxRenderer->getGLXContext();
    X11DisplayShPtr displayPtr = glxRenderer->getXDisplayPtr();

    CHECK(displayPtr == myDisplayPtr);

    // TODO(croot): do we need to do a flush/finish before copying?
    // or will the copy take care of that for us?

    auto& framebufferPtr = itr->second.getFramebuffer();
    CHECK(framebufferPtr);

    rgbaTex = framebufferPtr->getColorTexture2d(FboColorBuffer::COLOR_BUFFER);
    CHECK(rgbaTex);

    auto rgbaItr = _rgbaTextures.find(rgbaTex.get());
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
                          _rgbaTextureArray->getId(),
                          _rgbaTextureArray->getTarget(),
                          0,
                          0,
                          0,
                          cnt,
                          width,
                          height,
                          1);

    if (doHitTest) {
      idTex = framebufferPtr->getColorTexture2d(FboColorBuffer::ID_BUFFER);
      CHECK(idTex);

      auto idItr = _idTextures.find(idTex.get());
      CHECK(idItr != _idTextures.end());

      glXCopyImageSubDataNV(displayPtr.get(),
                            glxCtx,
                            idTex->getId(),
                            idTex->getTarget(),
                            0,
                            0,
                            0,
                            0,
                            myGlxCtx,
                            _idTextureArray->getId(),
                            _idTextureArray->getTarget(),
                            0,
                            0,
                            0,
                            cnt,
                            width,
                            height,
                            1);
    }

    // if (doDepthTest) {
    //   depthRbo = itr->second.framebufferPtr->getRenderbuffer(FboRenderBuffer::DEPTH_BUFFER);
    // }
  }

  // now do the composite pass
  // TODO(croo): add a concept of a "pass" to render
  _renderer->makeActiveOnCurrentThread();

  _framebufferPtr->bindToRenderer(_renderer);
  _renderer->bindShader(_shader);
  _renderer->bindVertexArray(_vao);

  // we're doing the blending manually in the shader, so disable any blending here
  _renderer->disable(GL_BLEND);

  _renderer->setViewport(0, 0, getWidth(), getHeight());
  _renderer->setClearColor(0, 0, 0, 0);
  _renderer->clearAll();

  _shader->setSamplerAttribute("rgbaArraySampler", _rgbaTextureArray);
  _shader->setUniformAttribute("rgbaArraySize", perGpuData->size());

  if (doHitTest) {
    _shader->setSamplerAttribute("idArraySampler", _idTextureArray);
    _shader->setUniformAttribute("idArraySize", perGpuData->size());
  } else {
    _shader->setUniformAttribute("idArraySize", 0);
  }

  _renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);
}

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer
