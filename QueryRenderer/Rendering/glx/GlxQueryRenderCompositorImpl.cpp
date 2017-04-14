#include "../../PngData.h"
#include "../../QueryRenderManager.h"
#include "../../QueryRendererContext.h"
#include "../../Scales/Scale.h"
#include <Rendering/Renderer/GL/glx/GlxGLRenderer.h>
#include "../QueryFramebuffer.h"
#include "GlxQueryRenderCompositorImpl.h"
#include "../shaders/multiGpuComposite_vert.h"
#include "../shaders/multiGpuComposite_frag.h"
#include "../shaders/multiGpuCompositeAccumulator_frag.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Renderer/GL/Resources/GLBufferLayout.h>
#include <Rendering/Renderer/GL/Resources/GLTexture2d.h>
#include <Rendering/Renderer/GL/glx/X11DisplayManager.h>

#include <boost/algorithm/string/replace.hpp>

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

const int GlxQueryRenderCompositorImpl::maxAccumColors = static_cast<int>(
    BaseScale::convertNumAccumTexturesToNumAccumVals(BaseScale::maxAccumTextures, AccumulatorType::BLEND));

GlxQueryRenderCompositorImpl::GlxQueryRenderCompositorImpl(QueryRenderManager* prnt,
                                                           ::Rendering::RendererShPtr& rendererPtr,
                                                           const size_t width,
                                                           const size_t height,
                                                           const size_t numSamples,
                                                           const bool doHitTest,
                                                           const bool doDepthTest,
                                                           const bool supportsInt64)
    : QueryRenderCompositorImpl(prnt, rendererPtr, width, height, numSamples, doHitTest, doDepthTest),
      _rendererPtr(rendererPtr) {
  CHECK(rendererPtr);
  _renderer = dynamic_cast<GlxGLRenderer*>(rendererPtr.get());
  CHECK(_renderer != nullptr);

  RUNTIME_EX_ASSERT(
      GLXEW_NV_copy_image,
      "NV_copy_image GLX extension is not supported. Cannot initialize compositor for multi-gpu rendering.");

  _initResources(prnt, supportsInt64);
}

void GlxQueryRenderCompositorImpl::_initResources(QueryRenderManager* queryRenderManager, const bool supportsInt64) {
  _renderer->makeActiveOnCurrentThread();
  GLResourceManagerShPtr rsrcMgr = _renderer->getResourceManager();

  // create a rectangle vertex buffer that will cover the entire buffer
  // and textured with the textures from all the gpus
  GLInterleavedBufferLayoutShPtr bufferLayout(new GLInterleavedBufferLayout(_renderer->getSupportedExtensions()));
  bufferLayout->addAttribute<float, 2>("pos");
  _rectvbo = rsrcMgr->createVertexBuffer<float>({-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, bufferLayout);

  std::string fragSrc(MultiGpuComposite_frag::source);
  bool doMultiSample = (getNumSamples() > 1);
  boost::replace_first(fragSrc, "<doMultiSample>", std::to_string(doMultiSample));
  boost::replace_first(fragSrc, "<maxAccumColors>", std::to_string(maxAccumColors));

  _shader = rsrcMgr->createShader(MultiGpuComposite_vert::source, fragSrc);
  _renderer->bindShader(_shader);

  // TODO(croot): automate the texture image unit binding
  _shader->setSamplerTextureImageUnit("rgbaArraySampler", GL_TEXTURE0);
  _shader->setSamplerTextureImageUnit("id1AArraySampler", GL_TEXTURE1);
  _shader->setSamplerTextureImageUnit("id1BArraySampler", GL_TEXTURE2);
  _shader->setSamplerTextureImageUnit("id2ArraySampler", GL_TEXTURE3);

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

  _rgbaTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_RGBA8, numSamples, sampleProps);

  if (doHitTest()) {
    _id1ATextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI, numSamples, sampleProps);
    if (supportsInt64) {
      _id1BTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI, numSamples, sampleProps);
    }
    _id2TextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32I, numSamples, sampleProps);
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

  if (_id1ATextureArray) {
    CHECK(doHitTest());
    _id1ATextureArray->resize(width, height);
  }

  if (_id1BTextureArray) {
    CHECK(doHitTest());
    _id1BTextureArray->resize(width, height);
  }

  if (_id2TextureArray) {
    CHECK(doHitTest());
    _id2TextureArray->resize(width, height);
  }

  // TODO(croot): do depth

  if (_accumulationCpTextureArray) {
    _accumulationCpTextureArray->resize(width, height);
  }

  if (_accumulationTextureArray) {
    _accumulationTextureArray->resize(width, height);
  }

  if (_clearPboPtr) {
    ::Rendering::Objects::Array2d<unsigned int> clearData(width, height, 0);
    _clearPboPtr->resize(width, height, clearData.getDataPtr());
  }
}

GLTexture2dShPtr GlxQueryRenderCompositorImpl::createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) {
  size_t width = getWidth();
  size_t height = getHeight();
  size_t numSamples = getNumSamples();

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLTexture2dShPtr tex = QueryFramebuffer::createFboTexture2d(rsrcMgr, texType, width, height, numSamples);

  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      _rgbaTextures.insert(tex.get());
      break;
    case FboColorBuffer::ID1A_BUFFER:
      _id1ATextures.insert(tex.get());
      break;
    case FboColorBuffer::ID1B_BUFFER:
      _id1BTextures.insert(tex.get());
      break;
    case FboColorBuffer::ID2_BUFFER:
      _id2Textures.insert(tex.get());
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
  size_t numSamples = getNumSamples();

  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

  GLRenderbufferShPtr rbo = QueryFramebuffer::createFboRenderbuffer(rsrcMgr, rboType, width, height, numSamples);

  _rbos.insert(rbo.get());

  return rbo;
}

void GlxQueryRenderCompositorImpl::_initAccumResources(size_t width, size_t height, size_t depth) {
  GLResourceManagerShPtr rsrcMgr = nullptr;
  CHECK(!_rendererPtr.expired());

  GLRenderer* currRenderer{nullptr};
  bool resetRenderer = false;

  if (!_accumulationCpTextureArray) {
    CHECK(!_accumulationTextureArray && !_accumulatorShader && !_clearPboPtr);

    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != _renderer))) {
      _renderer->makeActiveOnCurrentThread();
    }
    rsrcMgr = _renderer->getResourceManager();

    _accumulationCpTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI);
    _accumulationTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI);
    _accumulatorShader =
        rsrcMgr->createShader(MultiGpuComposite_vert::source, MultiGpuCompositeAccumulator_frag::source);
    ::Rendering::Objects::Array2d<unsigned int> clearData(width, height, 0);
    _clearPboPtr = rsrcMgr->createPixelBuffer2d(width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, clearData.getDataPtr());
  } else if (depth > _accumulationCpTextureArray->getDepth()) {
    _renderer->makeActiveOnCurrentThread();
    _accumulationCpTextureArray->resize(depth);
    _accumulationTextureArray->resize(depth);
  }

  CHECK(_accumulationCpTextureArray && _accumulationTextureArray && _accumulatorShader && _clearPboPtr &&
        _accumulationCpTextureArray->getWidth() == _accumulationTextureArray->getWidth() &&
        _accumulationCpTextureArray->getHeight() == _accumulationTextureArray->getHeight() &&
        _accumulationCpTextureArray->getDepth() == _accumulationTextureArray->getDepth());

  if (resetRenderer) {
    currRenderer->makeActiveOnCurrentThread();
  }
}

void GlxQueryRenderCompositorImpl::_cleanupAccumResources() {
  GLRenderer* currRenderer;
  bool resetRenderer = false;

  // TODO(croot): should we cleanup any expired weak ptrs?
  int maxSize = -1;
  for (int i = _registeredAccumTxts.size() - 1; i >= 0; --i) {
    if (_registeredAccumTxts[i].size() > 0) {
      maxSize = i;
      break;
    }
  }

  maxSize++;

  if (maxSize == 0) {
    CHECK(!_rendererPtr.expired());

    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != _renderer))) {
      _renderer->makeActiveOnCurrentThread();
    }

    _accumulationCpTextureArray = nullptr;
    _accumulationTextureArray = nullptr;
    _accumulatorShader = nullptr;
    _clearPboPtr = nullptr;
  } else if (maxSize < static_cast<int>(_registeredAccumTxts.size())) {
    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != _renderer))) {
      _renderer->makeActiveOnCurrentThread();
    }

    _accumulationCpTextureArray->resize(maxSize);
    _accumulationTextureArray->resize(maxSize);

    _registeredAccumTxts.resize(maxSize);
  }

  if (resetRenderer) {
    currRenderer->makeActiveOnCurrentThread();
  }
}

void GlxQueryRenderCompositorImpl::registerAccumulatorTexture(GLTexture2dShPtr& tex,
                                                              size_t accumIdx,
                                                              size_t numTexturesInArray) {
  // TODO(croot): make thread safe?

  CHECK(accumIdx < numTexturesInArray);

  _initAccumResources(getWidth(), getHeight(), numTexturesInArray);

  if (numTexturesInArray > _registeredAccumTxts.size()) {
    _registeredAccumTxts.resize(numTexturesInArray);
  }

  // TODO(croot): should we be careful and check to see if the tex has already been added?
  _registeredAccumTxts[accumIdx].emplace(tex.get(), tex);
}

void GlxQueryRenderCompositorImpl::unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                                                size_t accumIdx) {
  // TODO(croot): if the tex is not found at the accumIdx, should we indicate where it
  // can be found?

  // TODO(croot): make thread safe?

  CHECK(accumIdx < _registeredAccumTxts.size());

  auto& map = _registeredAccumTxts[accumIdx];
  auto itr = map.find(tex.get());
  CHECK(itr != map.end());

  map.erase(itr);

  _cleanupAccumResources();
}

void GlxQueryRenderCompositorImpl::unregisterAllAccumulatorTextures() {
  // TODO(croot): This should currently only be called when the server is shutting down,
  // But if that were to ever change, and be called in some other way, then we'll need to
  // clean up the proper resources here.
  //
  // Until then, this is a noop as this object's destructor will clean up the resources
}

void GlxQueryRenderCompositorImpl::_postPassPerGpuCB(::Rendering::GL::GLRenderer* renderer,
                                                     QueryFramebufferShPtr& framebufferPtr,
                                                     size_t width,
                                                     size_t height,
                                                     bool doHitTest,
                                                     bool doDepthTest,
                                                     int gpuCnt,
                                                     ScaleShPtr& accumulatorScalePtr,
                                                     int accumulatorCnt) {
  GLXContext myGlxCtx = _renderer->getGLXContext();
  X11DisplayShPtr myDisplayPtr = _renderer->getXDisplayPtr();

  GlxGLRenderer* glxRenderer = dynamic_cast<GlxGLRenderer*>(renderer);
  CHECK(glxRenderer != nullptr);

  GLXContext glxCtx = glxRenderer->getGLXContext();
  X11DisplayShPtr displayPtr = glxRenderer->getXDisplayPtr();

  CHECK(displayPtr == myDisplayPtr);

  // TODO(croot): do we need to do a flush/finish before copying?
  // or will the copy take care of that for us?

  if (accumulatorScalePtr) {
    CHECK(_accumulationCpTextureArray->getDepth() == _accumulationTextureArray->getDepth());

    auto& textures = accumulatorScalePtr->getAccumulatorTextureArrayRef(renderer->getGpuId());
    CHECK(textures.size() <= _accumulationCpTextureArray->getDepth());

    for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
      auto txWidth = textures[i]->getWidth();
      auto txHeight = textures[i]->getHeight();
      CHECK(txWidth <= _accumulationCpTextureArray->getWidth() &&
            txHeight <= _accumulationCpTextureArray->getHeight() && txWidth <= _accumulationTextureArray->getWidth() &&
            txHeight <= _accumulationTextureArray->getHeight() && width <= txWidth && height <= txHeight);
      glXCopyImageSubDataNV(displayPtr.get(),
                            glxCtx,
                            textures[i]->getId(),
                            textures[i]->getTarget(),
                            0,
                            0,
                            0,
                            0,
                            myGlxCtx,
                            _accumulationCpTextureArray->getId(),
                            _accumulationCpTextureArray->getTarget(),
                            0,
                            0,
                            0,
                            i,
                            width,
                            height,
                            1);
    }

    // TODO(croot): this extra pass here could be a bottleneck.
    // Decided to do it this way because otherwise we'd have to create
    // a texture array for each accumulator texture per gpu, which
    // could be a memory hog depending on how many gpus and how
    // many accumulator textures.
    _renderer->makeActiveOnCurrentThread();

    // clear out the accumulation texture the first time
    if (accumulatorCnt == 0 && gpuCnt == 0) {
      _accumulationTextureArray->copyPixelsFromPixelBuffer(_clearPboPtr);
    }

    _framebufferPtr->bindToRenderer(_renderer);
    _renderer->bindShader(_accumulatorShader);
    _accumulatorShader->setSamplerTextureImageUnit("accumulatorSampler", GL_TEXTURE0);
    _accumulatorShader->setSamplerAttribute("accumulatorSampler", _accumulationCpTextureArray);
    _accumulatorShader->setImageLoadStoreAttribute("inTxPixelCounter", _accumulationTextureArray);
    _accumulatorShader->setUniformAttribute("numAccumTextures", textures.size());
    _renderer->bindVertexArray(_vao);

    // we're doing the blending manually in the shader, so disable any blending here
    _renderer->disable(GL_BLEND);

    // no multisampling, so force-disable any sample shading
    _renderer->disable(GL_SAMPLE_SHADING);

    _renderer->setViewport(0, 0, width, height);

    _renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);
  } else {
    auto rgbaTex = framebufferPtr->getGLTexture2d(FboColorBuffer::COLOR_BUFFER);
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
                          gpuCnt,
                          width,
                          height,
                          1);
  }

  if (doHitTest) {
    auto id1ATex = framebufferPtr->getGLTexture2d(FboColorBuffer::ID1A_BUFFER);
    auto id1BTex = framebufferPtr->getGLTexture2d(FboColorBuffer::ID1B_BUFFER);
    auto id2Tex = framebufferPtr->getGLTexture2d(FboColorBuffer::ID2_BUFFER);
    CHECK(id1ATex && id2Tex);

    auto id1AItr = _id1ATextures.find(id1ATex.get());
    CHECK(id1AItr != _id1ATextures.end());

    auto id2Itr = _id2Textures.find(id2Tex.get());
    CHECK(id2Itr != _id2Textures.end());

    glXCopyImageSubDataNV(displayPtr.get(),
                          glxCtx,
                          id1ATex->getId(),
                          id1ATex->getTarget(),
                          0,
                          0,
                          0,
                          0,
                          myGlxCtx,
                          _id1ATextureArray->getId(),
                          _id1ATextureArray->getTarget(),
                          0,
                          0,
                          0,
                          gpuCnt,
                          width,
                          height,
                          1);

    if (id1BTex) {
      auto id1BItr = _id1BTextures.find(id1BTex.get());
      CHECK(id1BItr != _id1BTextures.end());
      glXCopyImageSubDataNV(displayPtr.get(),
                            glxCtx,
                            id1BTex->getId(),
                            id1BTex->getTarget(),
                            0,
                            0,
                            0,
                            0,
                            myGlxCtx,
                            _id1BTextureArray->getId(),
                            _id1BTextureArray->getTarget(),
                            0,
                            0,
                            0,
                            gpuCnt,
                            width,
                            height,
                            1);
    }

    glXCopyImageSubDataNV(displayPtr.get(),
                          glxCtx,
                          id2Tex->getId(),
                          id2Tex->getTarget(),
                          0,
                          0,
                          0,
                          0,
                          myGlxCtx,
                          _id2TextureArray->getId(),
                          _id2TextureArray->getTarget(),
                          0,
                          0,
                          0,
                          gpuCnt,
                          width,
                          height,
                          1);
  }

  // if (doDepthTest) {
  //   depthRbo = itr->second.framebufferPtr->getRenderbuffer(FboRenderBuffer::DEPTH_BUFFER);
  // }
}

void GlxQueryRenderCompositorImpl::_compositePass(const std::set<GpuId>& usedGpus,
                                                  size_t width,
                                                  size_t height,
                                                  bool doHitTest,
                                                  bool doDepthTest,
                                                  int passCnt,
                                                  ScaleShPtr& accumulatorScalePtr) {
  _renderer->makeActiveOnCurrentThread();

  _framebufferPtr->setHitTest(accumulatorScalePtr ? false : doHitTest);
  _framebufferPtr->setDepthTest(doDepthTest);
  _framebufferPtr->bindToRenderer(_renderer);

  _renderer->setViewport(0, 0, width, height);
  if (passCnt == 0) {
    _renderer->setClearColor(0, 0, 0, 0);
    _renderer->clearAll();
  }

  if (accumulatorScalePtr) {
    // NOTE: the blend equations should have been set in QueryRenderer
    _renderer->enable(GL_BLEND);
    _renderer->setBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    _renderer->setBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    accumulatorScalePtr->renderAccumulation(_renderer, _renderer->getGpuId(), _accumulationTextureArray.get());

    if (!doHitTest) {
      // can just return at this point
      // TODO(croot): this may need to change at some point if
      // other things need to be rendered with an accumulation
      // Depth? Alternate rendering styles (i.e. stroking?)
      return;
    }

    _framebufferPtr->setHitTest(doHitTest);
    _framebufferPtr->setDepthTest(doDepthTest);
    _framebufferPtr->bindToRenderer(_renderer);
  }

  _renderer->bindShader(_shader);
  _renderer->bindVertexArray(_vao);

  // we're doing the blending manually in the shader, so disable any blending here
  // TODO(croot): but what about multiple passes? Is this going to work then?
  // I think this will be a bug!!!!
  // _renderer->disable(GL_BLEND);

  // When multi-sampling, we want to read from all the samples too,
  // so enable sample shading to enforce that.
  _renderer->enable(GL_SAMPLE_SHADING);
  _renderer->setMinSampleShading(1.0);

  _shader->setSamplerAttribute("rgbaArraySampler", _rgbaTextureArray);
  _shader->setUniformAttribute("rgbaArraySize", usedGpus.size());

  if (doHitTest) {
    CHECK(_id1ATextureArray->getDepth() == _id2TextureArray->getDepth());
    _shader->setSamplerAttribute("id1AArraySampler", _id1ATextureArray);
    if (_id1BTextureArray) {
      CHECK(_id1ATextureArray->getDepth() == _id1BTextureArray->getDepth());
      _shader->setSamplerAttribute("id1BArraySampler", _id1BTextureArray);
    }
    _shader->setSamplerAttribute("id2ArraySampler", _id2TextureArray);
    _shader->setUniformAttribute("idArraySize", usedGpus.size());
    _shader->setUniformAttribute("idBArraySize", (_id1BTextureArray ? usedGpus.size() : 0));
  } else {
    _shader->setUniformAttribute("idArraySize", 0);
  }

  _shader->setUniformAttribute("passIdx", passCnt);

  _shader->setSubroutine("Compositor", (accumulatorScalePtr ? "compositeAccumulatorIds" : "compositeColor"));
  _renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);

  // TODO(croot): push/pop a state?
  _renderer->disable(GL_SAMPLE_SHADING);
}

void GlxQueryRenderCompositorImpl::render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) {
  QueryRendererContext* ctx = queryRenderer->getContext();

  auto qrmPerGpuDataPtr = ctx->getRootGpuCache()->perGpuData;

  bool doHitTest = ctx->doHitTest();
  // bool doDepthTest = ctx->doDepthTest();

  GLTexture2dShPtr rgbaTex, idTex, id2Tex;
  GLRenderbufferShPtr depthRbo;

  CHECK(_rgbaTextureArray && _rgbaTextures.size());
  if (_rgbaTextureArray->getDepth() != _rgbaTextures.size()) {
    _renderer->makeActiveOnCurrentThread();
    _rgbaTextureArray->resize(_rgbaTextures.size());
    CHECK(_rgbaTextures.size() == _rgbaTextureArray->getDepth());
  }

  if (doHitTest) {
    CHECK(_id1ATextureArray && _id2TextureArray && _id1ATextures.size() &&
          _id1ATextures.size() == _rgbaTextures.size() && _id2Textures.size() &&
          _id2Textures.size() == _rgbaTextures.size());
    if (_id1ATextureArray->getDepth() != _id1ATextures.size()) {
      _renderer->makeActiveOnCurrentThread();
      _id1ATextureArray->resize(_id1ATextures.size());
      CHECK(_id1ATextures.size() == _id1ATextureArray->getDepth());
    }

    if (_id1BTextureArray && _id1BTextureArray->getDepth() != _id1BTextures.size()) {
      CHECK(_id1BTextures.size() == _rgbaTextures.size());
      _renderer->makeActiveOnCurrentThread();
      _id1BTextureArray->resize(_id1BTextures.size());
      CHECK(_id1BTextures.size() == _id1BTextureArray->getDepth());
    }

    if (_id2TextureArray->getDepth() != _id2Textures.size()) {
      _renderer->makeActiveOnCurrentThread();
      _id2TextureArray->resize(_id2Textures.size());
      CHECK(_id2Textures.size() == _id2TextureArray->getDepth() && _id1ATextures.size() == _id2Textures.size());
    }
  }
  // if (doDepthTest) {
  //   CHECK(_rbos.size() == _)
  // }

  QueryRenderer::renderPasses(qrmPerGpuDataPtr,
                              ctx,
                              usedGpus,
                              true,
                              std::bind(&GlxQueryRenderCompositorImpl::_postPassPerGpuCB,
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
                              std::bind(&GlxQueryRenderCompositorImpl::_compositePass,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2,
                                        std::placeholders::_3,
                                        std::placeholders::_4,
                                        std::placeholders::_5,
                                        std::placeholders::_6,
                                        std::placeholders::_7));
}

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer
