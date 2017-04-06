#include "../../QueryRenderManager.h"
#include "../../QueryRenderer.h"
#include "../../QueryRendererContext.h"
#include "../../Scales/Scale.h"
#include <Rendering/Renderer/GL/egl/EglGLRenderer.h>
#include "../QueryFramebuffer.h"
#include "EglQueryRenderCompositorImpl.h"
#include "../shaders/multiGpuComposite_vert.h"
#include "../shaders/multiGpuComposite_frag.h"
#include "../shaders/multiGpuCompositeAccumulator_frag.h"
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
using ::Rendering::GL::Resources::GLInterleavedBufferLayout;
using ::Rendering::GL::Resources::GLInterleavedBufferLayoutShPtr;
using ::Rendering::GL::Resources::GLShaderShPtr;
using ::Rendering::GL::Resources::GLTexture2dArrayShPtr;
using ::Rendering::GL::Resources::GLVertexBufferShPtr;
using ::Rendering::GL::Resources::GLVertexArrayShPtr;

using ::Rendering::GL::Resources::GLTexture2dSampleProps;

#ifdef HAVE_CUDA
static void checkCudaErrors(CUresult result, const char* filename, int lineno, const GpuId gpuId) {
  if (result == CUDA_ERROR_INVALID_GRAPHICS_CONTEXT) {
    auto renderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    if (!renderer) {
      CHECK(false) << "CUDA error code=" << result << ". No gl context is set as current. " << filename << ":"
                   << lineno;
    } else {
      CHECK(false) << "CUDA error code=" << result << ". Current context is on gpu: " << renderer->getGpuId()
                   << ", but expecting it to be on " << gpuId << ". " << filename << ":" << lineno;
    }
  } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    LOG(ERROR) << "CUDA out of memory error on gpu: " << gpuId << " during EGL compositing. " << filename << ":"
               << lineno;
    throw Rendering::OutOfGpuMemoryError("Cuda out of memory error on gpu: " + std::to_string(gpuId) +
                                         " - CUDA error code =" + std::to_string(result));
  } else {
    CHECK(result == CUDA_SUCCESS) << "CUDA error code=" << result << ": " << filename << ":" << lineno;
  }
}

static void copyTextureToTextureArray(EglQueryRenderCompositorImpl::CudaStateGuard& cudaStateGuard,
                                      const CUcontext& srcCtx,
                                      const GpuId srcGpuId,
                                      CUgraphicsResource& srcTexRsrc,
                                      const CUcontext& dstCtx,
                                      const GpuId dstGpuId,
                                      CUgraphicsResource& dstTexArrayRsrc,
                                      const size_t dstTexArrayIdx,
                                      const size_t width,
                                      const size_t height,
                                      const size_t numElemBytes,
                                      const bool doAsync = true) {
  CUarray dstTexPtr;
  checkCudaErrors(cuGraphicsSubResourceGetMappedArray(&dstTexPtr, dstTexArrayRsrc, dstTexArrayIdx, 0),
                  __FILE__,
                  __LINE__,
                  dstGpuId);

  CUarray srcTexPtr;
  cudaStateGuard.mapCudaRsrc(srcTexRsrc);
  checkCudaErrors(cuGraphicsSubResourceGetMappedArray(&srcTexPtr, srcTexRsrc, 0, 0), __FILE__, __LINE__, srcGpuId);

  // TODO(croot):
  // Using cuArrayGetDescriptor(), should I verify that the width/height
  // and element/pixel type are correct?

  CUDA_MEMCPY3D_PEER copyParams;
  copyParams.Depth = 1;
  copyParams.Height = height;
  copyParams.WidthInBytes = width * numElemBytes;
  copyParams.dstArray = dstTexPtr;
  copyParams.dstContext = dstCtx;
  copyParams.dstDevice = 0;
  copyParams.dstHeight = height;
  copyParams.dstHost = 0;
  copyParams.dstLOD = 0;
  copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyParams.dstPitch = 0;
  copyParams.dstXInBytes = 0;
  copyParams.dstY = 0;
  copyParams.dstZ = 0;
  copyParams.srcArray = srcTexPtr;
  copyParams.srcContext = srcCtx;
  copyParams.srcDevice = 0;
  copyParams.srcHeight = height;
  copyParams.srcHost = 0;
  copyParams.srcLOD = 0;
  copyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copyParams.srcPitch = 0;
  copyParams.srcXInBytes = 0;
  copyParams.srcY = 0;
  copyParams.srcZ = 0;

  if (doAsync) {
    checkCudaErrors(cuMemcpy3DPeerAsync(&copyParams, 0), __FILE__, __LINE__, srcGpuId);
  } else {
    checkCudaErrors(cuMemcpy3DPeer(&copyParams), __FILE__, __LINE__, srcGpuId);
  }
}

static CUgraphicsResource initCudaGraphicsResource(const GpuId gpuId,
                                                   const GLuint glTexId,
                                                   const GLenum glTarget,
                                                   unsigned int usageFlags = CU_GRAPHICS_REGISTER_FLAGS_NONE) {
  CUdevice ctxDevice;

  checkCudaErrors(cuCtxGetDevice(&ctxDevice), __FILE__, __LINE__, gpuId);

  RUNTIME_EX_ASSERT(ctxDevice == static_cast<int>(gpuId),
                    "Invalid cuda context when registering a texture for egl compositing. GL resource gpu id: " +
                        std::to_string(gpuId) + " does not match cuda device id: " + std::to_string(ctxDevice) + ".");

  CUgraphicsResource rsrc;
  auto result = cuGraphicsGLRegisterImage(&rsrc, glTexId, glTarget, usageFlags);

  if (result == CUDA_ERROR_UNKNOWN) {
    throw ::Rendering::OutOfGpuMemoryError("Cuda error code=" + std::to_string(result) +
                                           ", CUDA_ERROR_UNKNOWN, possibly not enough gpu memory available on gpu " +
                                           std::to_string(gpuId) + " for the requested texture id " +
                                           std::to_string(glTexId) + ".");
  }

  checkCudaErrors(result, __FILE__, __LINE__, gpuId);
  return rsrc;
}

static void destroyCudaGraphicsResource(const GpuId gpuId, CUgraphicsResource& cudaRsrc) {
  CUdevice ctxDevice;

  auto result = cuCtxGetDevice(&ctxDevice);
  if (result != CUDA_ERROR_DEINITIALIZED) {
    checkCudaErrors(result, __FILE__, __LINE__, gpuId);

    RUNTIME_EX_ASSERT(
        ctxDevice == static_cast<int>(gpuId),
        "Invalid cuda context when unregistering a texture used for egl compositing. GL resource gpu id: " +
            std::to_string(gpuId) + " does not match cuda device id: " + std::to_string(ctxDevice) + ".");

    checkCudaErrors(cuGraphicsUnregisterResource(cudaRsrc), __FILE__, __LINE__, gpuId);
  }
}
#endif

EglQueryRenderCompositorImpl::CudaStateGuard::CudaStateGuard(EglQueryRenderCompositorImpl* currCompositor)
    : gpuId(currCompositor->_rendererPtr->getGpuId()) {
#ifdef HAVE_CUDA
  auto gpuIdx = gpuId - currCompositor->_startGpuId;
  auto& cudaCtxVector = currCompositor->_cudaMgr->getDeviceContexts();
  ctx = cudaCtxVector[gpuIdx];
  checkCudaErrors(cuCtxPushCurrent(ctx), __FILE__, __LINE__, gpuId);
#endif
}

EglQueryRenderCompositorImpl::CudaStateGuard::CudaStateGuard(const CUcontext& ctx, const GpuId gpuId)
    : gpuId(gpuId), ctx(ctx) {
#ifdef HAVE_CUDA
  checkCudaErrors(cuCtxPushCurrent(ctx), __FILE__, __LINE__, gpuId);
#endif
}

EglQueryRenderCompositorImpl::CudaStateGuard::~CudaStateGuard() {
#ifdef HAVE_CUDA
  unmapAllCudaRsrcs();
  checkCudaErrors(cuCtxPopCurrent(0), __FILE__, __LINE__, gpuId);
#endif
}

void EglQueryRenderCompositorImpl::CudaStateGuard::mapCudaRsrc(CUgraphicsResource& cudaRsrc) {
#ifdef HAVE_CUDA
  if (mappedCudaRsrcs.find(cudaRsrc) == mappedCudaRsrcs.end()) {
    std::unique_ptr<CudaStateGuard> newGuard;  // used if we need to reset the cuda context
    CUdevice currDev;
    checkCudaErrors(cuCtxGetDevice(&currDev), __FILE__, __LINE__, gpuId);
    if (currDev != static_cast<int>(gpuId)) {
      newGuard.reset(new CudaStateGuard(ctx, gpuId));
    }

    checkCudaErrors(cuGraphicsMapResources(1, &cudaRsrc, 0), __FILE__, __LINE__, gpuId);
    mappedCudaRsrcs.insert(cudaRsrc);
  }
#endif
}

void EglQueryRenderCompositorImpl::CudaStateGuard::mapCudaRsrcs(std::vector<CUgraphicsResource>& cudaRsrcs) {
#ifdef HAVE_CUDA
  if (cudaRsrcs.size()) {
    std::unique_ptr<CudaStateGuard> newGuard;  // used if we need to reset the cuda context
    CUdevice currDev;
    checkCudaErrors(cuCtxGetDevice(&currDev), __FILE__, __LINE__, gpuId);
    if (currDev != static_cast<int>(gpuId)) {
      newGuard.reset(new CudaStateGuard(ctx, gpuId));
    }

    for (auto& cudaRsrc : cudaRsrcs) {
      if (mappedCudaRsrcs.find(cudaRsrc) == mappedCudaRsrcs.end()) {
        checkCudaErrors(cuGraphicsMapResources(1, &cudaRsrc, 0), __FILE__, __LINE__, gpuId);
        mappedCudaRsrcs.insert(cudaRsrc);
      }
    }
  }
#endif
}

void EglQueryRenderCompositorImpl::CudaStateGuard::unmapCudaRsrc(CUgraphicsResource& cudaRsrc) {
#ifdef HAVE_CUDA
  auto itr = mappedCudaRsrcs.find(cudaRsrc);
  if (itr != mappedCudaRsrcs.end()) {
    std::unique_ptr<CudaStateGuard> newGuard;  // used if we need to reset the cuda context
    CUdevice currDev;
    checkCudaErrors(cuCtxGetDevice(&currDev), __FILE__, __LINE__, gpuId);
    if (currDev != static_cast<int>(gpuId)) {
      newGuard.reset(new CudaStateGuard(ctx, gpuId));
    }

    checkCudaErrors(cuGraphicsUnmapResources(1, &cudaRsrc, 0), __FILE__, __LINE__, gpuId);
    mappedCudaRsrcs.erase(itr);
  }
#endif
}

void EglQueryRenderCompositorImpl::CudaStateGuard::unmapAllCudaRsrcs() {
#ifdef HAVE_CUDA
  size_t cnt = 0;
  for (auto rsrc : mappedCudaRsrcs) {
    checkCudaErrors(cuGraphicsUnmapResources(1, &rsrc, 0), __FILE__, __LINE__, gpuId);
    cnt++;
  }
  mappedCudaRsrcs.clear();
#endif
}

const int EglQueryRenderCompositorImpl::maxAccumColors = static_cast<int>(
    BaseScale::convertNumAccumTexturesToNumAccumVals(BaseScale::maxAccumTextures, AccumulatorType::BLEND));

EglQueryRenderCompositorImpl::EglQueryRenderCompositorImpl(QueryRenderManager* prnt,
                                                           ::Rendering::RendererShPtr& rendererPtr,
                                                           const size_t width,
                                                           const size_t height,
                                                           const size_t numSamples,
                                                           const bool doHitTest,
                                                           const bool doDepthTest,
                                                           const bool supportsInt64)
    : QueryRenderCompositorImpl(prnt, rendererPtr, width, height, numSamples, doHitTest, doDepthTest),
      _startGpuId(prnt->getStartGpuId()),
      _rebuildCudaConsumedRsrcs(false) {
#ifdef HAVE_CUDA
  _cudaMgr = prnt->getCudaMgr();
  _rendererPtr = std::dynamic_pointer_cast<GLRenderer>(rendererPtr);
  CHECK(_rendererPtr);

  _initResources(prnt, supportsInt64);
#endif
}

void EglQueryRenderCompositorImpl::_initResources(QueryRenderManager* queryRenderManager, const bool supportsInt64) {
#ifdef HAVE_CUDA
  auto gpuId = _rendererPtr->getGpuId();
  _rendererPtr->makeActiveOnCurrentThread();
  GLResourceManagerShPtr rsrcMgr = _rendererPtr->getResourceManager();

  // create a rectangle vertex buffer that will cover the entire buffer
  // and textured with the textures from all the gpus
  GLInterleavedBufferLayoutShPtr bufferLayout(new GLInterleavedBufferLayout(_rendererPtr->getSupportedExtensions()));
  bufferLayout->addAttribute<float, 2>("pos");
  _rectvbo = rsrcMgr->createVertexBuffer<float>({-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, bufferLayout);

  std::string fragSrc(MultiGpuComposite_frag::source);
  bool doMultiSample = (getNumSamples() > 1);
  boost::replace_first(fragSrc, "<doMultiSample>", std::to_string(doMultiSample));
  boost::replace_first(fragSrc, "<maxAccumColors>", std::to_string(maxAccumColors));

  _shader = rsrcMgr->createShader(MultiGpuComposite_vert::source, fragSrc);
  _rendererPtr->bindShader(_shader);

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
  _rgbaRsrc = initCudaGraphicsResource(
      gpuId, _rgbaTextureArray->getId(), _rgbaTextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

  if (doHitTest()) {
    _id1ATextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI, numSamples, sampleProps);
    _id1ARsrc = initCudaGraphicsResource(
        gpuId, _id1ATextureArray->getId(), _id1ATextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    if (supportsInt64) {
      _id1BTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI, numSamples, sampleProps);
      _id1BRsrc = initCudaGraphicsResource(
          gpuId, _id1BTextureArray->getId(), _id1BTextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    }
    _id2TextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32I, numSamples, sampleProps);
    _id2Rsrc = initCudaGraphicsResource(
        gpuId, _id2TextureArray->getId(), _id2TextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  }
#endif
}

EglQueryRenderCompositorImpl::~EglQueryRenderCompositorImpl() {
  // need to make the renderer active to properly delete the GL resources
  // TODO(croot): reset to previously active renderer?
  _rendererPtr->makeActiveOnCurrentThread();
}

void EglQueryRenderCompositorImpl::_resizeImpl(size_t width, size_t height) {
#ifdef HAVE_CUDA
  auto gpuId = _rendererPtr->getGpuId();

  auto prevRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
  _rendererPtr->makeActiveOnCurrentThread();

  CudaStateGuard cudaStateGuard(this);
  if (_rgbaTextureArray) {
    destroyCudaGraphicsResource(gpuId, _rgbaRsrc);
    _rgbaTextureArray->resize(width, height);
    _rgbaRsrc = initCudaGraphicsResource(
        gpuId, _rgbaTextureArray->getId(), _rgbaTextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  }

  if (_id1ATextureArray) {
    CHECK(doHitTest());
    destroyCudaGraphicsResource(gpuId, _id1ARsrc);
    _id1ATextureArray->resize(width, height);
    _id1ARsrc = initCudaGraphicsResource(
        gpuId, _id1ATextureArray->getId(), _id1ATextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  }

  if (_id1BTextureArray) {
    CHECK(doHitTest());
    destroyCudaGraphicsResource(gpuId, _id1BRsrc);
    _id1BTextureArray->resize(width, height);
    _id1BRsrc = initCudaGraphicsResource(
        gpuId, _id1BTextureArray->getId(), _id1BTextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  }

  if (_id2TextureArray) {
    CHECK(doHitTest());
    destroyCudaGraphicsResource(gpuId, _id2Rsrc);
    _id2TextureArray->resize(width, height);
    _id2Rsrc = initCudaGraphicsResource(
        gpuId, _id2TextureArray->getId(), _id2TextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  }

  // TODO(croot): do depth

  if (_accumulationCpTextureArray) {
    destroyCudaGraphicsResource(gpuId, _accumulationCpRsrc);
    _accumulationCpTextureArray->resize(width, height);
    _accumulationCpRsrc = initCudaGraphicsResource(gpuId,
                                                   _accumulationCpTextureArray->getId(),
                                                   _accumulationCpTextureArray->getTarget(),
                                                   CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
  }

  if (_accumulationTextureArray) {
    _accumulationTextureArray->resize(width, height);
  }

  if (_clearPboPtr) {
    ::Rendering::Objects::Array2d<unsigned int> clearData(width, height, 0);
    _clearPboPtr->resize(width, height, clearData.getDataPtr());
  }

  if (!_rebuildCudaConsumedRsrcs) {
    auto qrmPerGpuDataPtr = _renderManager->getPerGpuData();
    auto& cudaCtxVector = _cudaMgr->getDeviceContexts();
    for (auto& itr : _consumedRsrcs) {
      auto perGpuItem = qrmPerGpuDataPtr->find(itr.first);
      CHECK(perGpuItem != qrmPerGpuDataPtr->end());
      (*perGpuItem)->makeActiveOnCurrentThread();

      auto gpuIdx = itr.first - _startGpuId;
      CudaStateGuard cudaStateGuard(cudaCtxVector[gpuIdx], itr.first);

      CUgraphicsResource tmp;
      for (auto& rsrc : itr.second.rgbaTextures) {
        tmp = rsrc.txCudaRsrc;
        destroyCudaGraphicsResource(itr.first, tmp);
      }

      for (auto& rsrc : itr.second.id1ATextures) {
        tmp = rsrc.txCudaRsrc;
        destroyCudaGraphicsResource(itr.first, tmp);
      }

      for (auto& rsrc : itr.second.id1BTextures) {
        tmp = rsrc.txCudaRsrc;
        destroyCudaGraphicsResource(itr.first, tmp);
      }

      for (auto& rsrc : itr.second.id2Textures) {
        tmp = rsrc.txCudaRsrc;
        destroyCudaGraphicsResource(itr.first, tmp);
      }

      // TODO(croot): handle depth
    }
    _rebuildCudaConsumedRsrcs = true;
  }

  if (prevRenderer) {
    prevRenderer->makeActiveOnCurrentThread();
  } else {
    auto currRenderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    currRenderer->makeInactive();
  }
#endif
}

GLTexture2dShPtr EglQueryRenderCompositorImpl::createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) {
  size_t width = getWidth();
  size_t height = getHeight();
  size_t numSamples = getNumSamples();

  auto gpuId = renderer->getGpuId();
  GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
  GLTexture2dShPtr tex = QueryFramebuffer::createFboTexture2d(rsrcMgr, texType, width, height, numSamples);

  auto itr = _consumedRsrcs.find(gpuId);
  if (itr == _consumedRsrcs.end()) {
    auto rtn = _consumedRsrcs.emplace(std::piecewise_construct, std::forward_as_tuple(gpuId), std::forward_as_tuple());
    CHECK(rtn.second);
    itr = rtn.first;
  }

  switch (texType) {
    case FboColorBuffer::COLOR_BUFFER:
      itr->second.rgbaTextures.emplace(tex);
      break;
    case FboColorBuffer::ID1A_BUFFER:
      itr->second.id1ATextures.emplace(tex);
      break;
    case FboColorBuffer::ID1B_BUFFER:
      itr->second.id1BTextures.emplace(tex);
      break;
    case FboColorBuffer::ID2_BUFFER:
      itr->second.id2Textures.emplace(tex);
      break;
    default:
      CHECK(false);
  }

  _rebuildCudaConsumedRsrcs = true;

  return tex;
}

GLRenderbufferShPtr EglQueryRenderCompositorImpl::createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) {
  THROW_RUNTIME_EX("Depth buffers are not supported yet");
  return nullptr;
}

void EglQueryRenderCompositorImpl::_initAccumResources(size_t width, size_t height, size_t depth) {
#ifdef HAVE_CUDA
  GLResourceManagerShPtr rsrcMgr = nullptr;
  GLRenderer* currRenderer;
  bool resetRenderer = false;

  if (!_accumulationCpTextureArray) {
    CHECK(!_accumulationTextureArray && !_accumulatorShader && !_clearPboPtr);

    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != _rendererPtr.get()))) {
      _rendererPtr->makeActiveOnCurrentThread();
    }
    rsrcMgr = _rendererPtr->getResourceManager();

    _accumulationCpTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI);
    _accumulationCpRsrc = initCudaGraphicsResource(_rendererPtr->getGpuId(),
                                                   _accumulationCpTextureArray->getId(),
                                                   _accumulationCpTextureArray->getTarget(),
                                                   CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    _accumulationTextureArray = rsrcMgr->createTexture2dArray(width, height, depth, GL_R32UI);
    _accumulatorShader =
        rsrcMgr->createShader(MultiGpuComposite_vert::source, MultiGpuCompositeAccumulator_frag::source);
    ::Rendering::Objects::Array2d<unsigned int> clearData(width, height, 0);
    _clearPboPtr = rsrcMgr->createPixelBuffer2d(width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, clearData.getDataPtr());
  } else if (depth > _accumulationCpTextureArray->getDepth()) {
    auto gpuId = _rendererPtr->getGpuId();
    _rendererPtr->makeActiveOnCurrentThread();

    destroyCudaGraphicsResource(gpuId, _accumulationCpRsrc);
    _accumulationCpTextureArray->resize(depth);
    _accumulationCpRsrc = initCudaGraphicsResource(_rendererPtr->getGpuId(),
                                                   _accumulationCpTextureArray->getId(),
                                                   _accumulationCpTextureArray->getTarget(),
                                                   CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

    _accumulationTextureArray->resize(depth);
  }

  CHECK(_accumulationCpTextureArray && _accumulationTextureArray && _accumulatorShader && _clearPboPtr &&
        _accumulationCpTextureArray->getWidth() == _accumulationTextureArray->getWidth() &&
        _accumulationCpTextureArray->getHeight() == _accumulationTextureArray->getHeight() &&
        _accumulationCpTextureArray->getDepth() == _accumulationTextureArray->getDepth());

  if (resetRenderer) {
    currRenderer->makeActiveOnCurrentThread();
  }
#endif
}

void EglQueryRenderCompositorImpl::_cleanupAccumResources() {
#ifdef HAVE_CUDA
  GLRenderer* currRenderer;
  bool resetRenderer = false;

  // TODO(croot): should we cleanup any expired weak ptrs?
  size_t maxSize = 0;
  for (auto& itr : _consumedRsrcs) {
    for (int i = static_cast<int>(itr.second.accumTextures.size()) - 1; i >= static_cast<int>(maxSize); --i) {
      if (!itr.second.accumTextures[i].isEmpty() && i > static_cast<int>(maxSize)) {
        maxSize = i + 1;
        break;
      }
    }
  }

  if (maxSize == 0) {
    currRenderer = GLRenderer::getCurrentThreadRenderer();
    if ((resetRenderer = (currRenderer != _rendererPtr.get()))) {
      _rendererPtr->makeActiveOnCurrentThread();
    }

    auto gpuId = _rendererPtr->getGpuId();
    auto gpuIdx = gpuId - _startGpuId;
    auto& cudaCtxVector = _cudaMgr->getDeviceContexts();
    CudaStateGuard cudaStateGuard(cudaCtxVector[gpuIdx], gpuId);

    destroyCudaGraphicsResource(gpuId, _accumulationCpRsrc);
    _accumulationCpTextureArray = nullptr;

    _accumulationTextureArray = nullptr;
    _accumulatorShader = nullptr;
    _clearPboPtr = nullptr;
  } else {
    CHECK(_accumulationCpTextureArray && _accumulationTextureArray &&
          _accumulationCpTextureArray->getDepth() == _accumulationTextureArray->getDepth())
        << "accumulation cp texture array size: "
        << (_accumulationCpTextureArray ? _accumulationTextureArray->getDepth() : 0)
        << ", final accumulation texture array size: "
        << (_accumulationTextureArray ? _accumulationTextureArray->getDepth() : 0);

    currRenderer = GLRenderer::getCurrentThreadRenderer();
    auto& cudaCtxVector = _cudaMgr->getDeviceContexts();
    if (maxSize != _accumulationCpTextureArray->getDepth()) {
      if ((resetRenderer = (currRenderer != _rendererPtr.get()))) {
        _rendererPtr->makeActiveOnCurrentThread();
      }

      auto gpuId = _rendererPtr->getGpuId();
      auto gpuIdx = gpuId - _startGpuId;
      CudaStateGuard cudaStateGuard(cudaCtxVector[gpuIdx], gpuId);
      destroyCudaGraphicsResource(gpuId, _accumulationCpRsrc);
      _accumulationCpTextureArray->resize(maxSize);
      _accumulationCpRsrc = initCudaGraphicsResource(gpuId,
                                                     _accumulationCpTextureArray->getId(),
                                                     _accumulationCpTextureArray->getTarget(),
                                                     CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

      _accumulationTextureArray->resize(maxSize);
    }
  }

  if (resetRenderer) {
    currRenderer->makeActiveOnCurrentThread();
  }
#endif
}

void EglQueryRenderCompositorImpl::registerAccumulatorTexture(GLTexture2dShPtr& tex,
                                                              size_t accumIdx,
                                                              size_t numTexturesInArray) {
  // TODO(croot): make thread safe?

  CHECK(accumIdx < numTexturesInArray);

  _initAccumResources(getWidth(), getHeight(), numTexturesInArray);

  auto gpuId = tex->getGLRenderer()->getGpuId();

  auto itr = _consumedRsrcs.find(gpuId);
  if (itr == _consumedRsrcs.end()) {
    auto rtn = _consumedRsrcs.emplace(std::piecewise_construct, std::forward_as_tuple(gpuId), std::forward_as_tuple());
    CHECK(rtn.second);
    itr = rtn.first;
  }

  for (auto& curritr : _consumedRsrcs) {
    if (numTexturesInArray > curritr.second.accumTextures.size()) {
      curritr.second.accumTextures.resize(numTexturesInArray);
    }
  }

  // TODO(croot): should we be careful and check to see if the tex has already been added?
  itr->second.accumTextures[accumIdx].set(tex);
}

void EglQueryRenderCompositorImpl::unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                                                size_t accumIdx) {
  // TODO(croot): if the tex is not found at the accumIdx, should we indicate where it
  // can be found?

  // TODO(croot): make thread safe?

  auto gpuId = tex->getGLRenderer()->getGpuId();
  auto itr = _consumedRsrcs.find(gpuId);
  CHECK(itr != _consumedRsrcs.end() && accumIdx < itr->second.accumTextures.size() &&
        itr->second.accumTextures[accumIdx].txPtr == tex.get());

  itr->second.accumTextures[accumIdx].empty();
  _cleanupAccumResources();
}

void EglQueryRenderCompositorImpl::unregisterAllAccumulatorTextures() {
  // TODO(croot): This should currently only be called when the server is shutting down,
  // so we shouldn't have to worry about unregistering the cuda graphics resources.
  // But if we need to worry about proper unregistering of the cuda graphics resource,
  // even on shutdown, then we need to make the _cudaMtr pointer a shared resource
  // as that can otherwise be destroyed before we reach here and results in a segfault
  // when trying to access it in some way.

  // So this is a noop because this object's destructor will take care of cleanup.
}

void EglQueryRenderCompositorImpl::_postPassPerGpuCB(::Rendering::GL::GLRenderer* renderer,
                                                     QueryFramebufferShPtr& framebufferPtr,
                                                     size_t width,
                                                     size_t height,
                                                     bool doHitTest,
                                                     bool doDepthTest,
                                                     int gpuCnt,
                                                     ScaleShPtr& accumulatorScalePtr,
                                                     int accumulatorCnt) {
#ifdef HAVE_CUDA
  // TODO(croot): do we need to do a flush/finish before copying?
  // or will the copy take care of that for us?

  // NOTE: the cuda context for the gpu this composite is going to run on
  // should've been made current already

  auto srcGpuId = renderer->getGpuId();
  auto dstGpuId = _rendererPtr->getGpuId();
  auto srcGpuIdx = srcGpuId - _startGpuId;
  auto dstGpuIdx = dstGpuId - _startGpuId;

  auto rsrcItr = _consumedRsrcs.find(srcGpuId);
  CHECK(rsrcItr != _consumedRsrcs.end());

  auto& cudaCtxVector = _cudaMgr->getDeviceContexts();
  std::unique_ptr<CudaStateGuard> localCudaStateGuard;

  if (accumulatorScalePtr) {
    CHECK(_accumulationCpTextureArray->getDepth() == _accumulationTextureArray->getDepth());
    _rendererPtr->makeActiveOnCurrentThread();
    _rndrCudaStateGuard->mapCudaRsrc(_accumulationCpRsrc);

    renderer->makeActiveOnCurrentThread();
    auto& textures = accumulatorScalePtr->getAccumulatorTextureArrayRef(renderer->getGpuId());
    CHECK(textures.size() <= _accumulationCpTextureArray->getDepth());

    CHECK(textures.size() <= rsrcItr->second.accumTextures.size());

    localCudaStateGuard.reset(new CudaStateGuard(cudaCtxVector[srcGpuIdx], srcGpuId));
    for (int i = 0; i < static_cast<int>(textures.size()); ++i) {
      auto txWidth = textures[i]->getWidth();
      auto txHeight = textures[i]->getHeight();
      CHECK(txWidth <= _accumulationCpTextureArray->getWidth() &&
            txHeight <= _accumulationCpTextureArray->getHeight() && txWidth <= _accumulationTextureArray->getWidth() &&
            txHeight <= _accumulationTextureArray->getHeight() && width <= txWidth && height <= txHeight);
      CHECK(rsrcItr->second.accumTextures[i].txPtr == textures[i].get());

      // NOTE: we're initializing and destroying the cuda graphics resource
      // for the accumulation count textures as they can be routinely resized,
      // and they need to be unregistered during resizing.
      // We don't know coming in whether those textures are going to be resized,
      // so we need to remove that dependency.
      // TODO(croot): to avert this init/destroy loop here, we could create a
      // QueryTexture object, much like QueryBuffer, that handles the CUDA mapping
      // and remove that resource management from here.
      rsrcItr->second.accumTextures[i].txCudaRsrc = initCudaGraphicsResource(
          srcGpuId, textures[i]->getId(), textures[i]->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);

      copyTextureToTextureArray(*localCudaStateGuard,
                                cudaCtxVector[srcGpuIdx],
                                srcGpuId,
                                rsrcItr->second.accumTextures[i].txCudaRsrc,
                                cudaCtxVector[dstGpuIdx],
                                dstGpuId,
                                _accumulationCpRsrc,
                                i,
                                width,
                                height,
                                sizeof(unsigned int),
                                false);

      localCudaStateGuard->unmapCudaRsrc(rsrcItr->second.accumTextures[i].txCudaRsrc);
      destroyCudaGraphicsResource(srcGpuId, rsrcItr->second.accumTextures[i].txCudaRsrc);
    }
    localCudaStateGuard = nullptr;

    // TODO(croot): this extra pass here could be a bottleneck.
    // Decided to do it this way because otherwise we'd have to create
    // a texture array for each accumulator texture per gpu, which
    // could be a memory hog depending on how many gpus and how
    // many accumulator textures.
    _rendererPtr->makeActiveOnCurrentThread();
    _rndrCudaStateGuard->unmapCudaRsrc(_accumulationCpRsrc);

    // clear out the accumulation texture the first time
    if (accumulatorCnt == 0 && gpuCnt == 0) {
      _accumulationTextureArray->copyPixelsFromPixelBuffer(_clearPboPtr);
    }

    _framebufferPtr->bindToRenderer(_rendererPtr.get());
    _rendererPtr->bindShader(_accumulatorShader);
    _accumulatorShader->setSamplerTextureImageUnit("accumulatorSampler", GL_TEXTURE0);
    _accumulatorShader->setSamplerAttribute("accumulatorSampler", _accumulationCpTextureArray);
    _accumulatorShader->setImageLoadStoreAttribute("inTxPixelCounter", _accumulationTextureArray);
    _accumulatorShader->setUniformAttribute("numAccumTextures", textures.size());
    _rendererPtr->bindVertexArray(_vao);

    // we're doing the blending manually in the shader, so disable any blending here
    _rendererPtr->disable(GL_BLEND);

    // no multisampling, so force-disable any sample shading
    _rendererPtr->disable(GL_SAMPLE_SHADING);

    _rendererPtr->setViewport(0, 0, width, height);

    _rendererPtr->drawVertexBuffers(GL_TRIANGLE_STRIP);
  } else {
    _rendererPtr->makeActiveOnCurrentThread();
    _rndrCudaStateGuard->mapCudaRsrc(_rgbaRsrc);

    renderer->makeActiveOnCurrentThread();
    localCudaStateGuard.reset(new CudaStateGuard(cudaCtxVector[srcGpuIdx], srcGpuId));
    auto rgbaTex = framebufferPtr->getGLTexture2d(FboColorBuffer::COLOR_BUFFER);
    CHECK(rgbaTex);

    auto rgbaRsrc = rsrcItr->second.rgbaTextures.find(rgbaTex.get());
    CHECK(rgbaRsrc != rsrcItr->second.rgbaTextures.end());
    auto rgbaCudaRsrc = rgbaRsrc->txCudaRsrc;

    copyTextureToTextureArray(*localCudaStateGuard,
                              cudaCtxVector[srcGpuIdx],
                              srcGpuId,
                              rgbaCudaRsrc,
                              cudaCtxVector[dstGpuIdx],
                              dstGpuId,
                              _rgbaRsrc,
                              gpuCnt,
                              width,
                              height,
                              4);
  }

  if (doHitTest) {
    auto id1ATex = framebufferPtr->getGLTexture2d(FboColorBuffer::ID1A_BUFFER);
    auto id1BTex = framebufferPtr->getGLTexture2d(FboColorBuffer::ID1B_BUFFER);
    auto id2Tex = framebufferPtr->getGLTexture2d(FboColorBuffer::ID2_BUFFER);
    CHECK(id1ATex && id2Tex);

    std::vector<CUgraphicsResource> rsrcs = {_id1ARsrc, _id2Rsrc};
    if (id1BTex) {
      rsrcs.push_back(_id1BRsrc);
    }
    _rendererPtr->makeActiveOnCurrentThread();
    _rndrCudaStateGuard->mapCudaRsrcs(rsrcs);

    renderer->makeActiveOnCurrentThread();
    if (!localCudaStateGuard) {
      localCudaStateGuard.reset(new CudaStateGuard(cudaCtxVector[srcGpuIdx], srcGpuId));
    }

    auto id1ARsrc = rsrcItr->second.id1ATextures.find(id1ATex.get());
    CHECK(id1ARsrc != rsrcItr->second.id1ATextures.end());
    auto id1ACudaRsrc = id1ARsrc->txCudaRsrc;

    copyTextureToTextureArray(*localCudaStateGuard,
                              cudaCtxVector[srcGpuIdx],
                              srcGpuId,
                              id1ACudaRsrc,
                              cudaCtxVector[dstGpuIdx],
                              dstGpuId,
                              _id1ARsrc,
                              gpuCnt,
                              width,
                              height,
                              sizeof(unsigned int));

    if (id1BTex) {
      auto id1BRsrc = rsrcItr->second.id1BTextures.find(id1BTex.get());
      CHECK(id1BRsrc != rsrcItr->second.id1BTextures.end());
      auto id1BCudaRsrc = id1BRsrc->txCudaRsrc;
      copyTextureToTextureArray(*localCudaStateGuard,
                                cudaCtxVector[srcGpuIdx],
                                srcGpuId,
                                id1BCudaRsrc,
                                cudaCtxVector[dstGpuIdx],
                                dstGpuId,
                                _id1BRsrc,
                                gpuCnt,
                                width,
                                height,
                                sizeof(unsigned int));
    }

    auto id2Rsrc = rsrcItr->second.id2Textures.find(id2Tex.get());
    CHECK(id2Rsrc != rsrcItr->second.id2Textures.end());
    auto id2CudaRsrc = id2Rsrc->txCudaRsrc;

    copyTextureToTextureArray(*localCudaStateGuard,
                              cudaCtxVector[srcGpuIdx],
                              srcGpuId,
                              id2CudaRsrc,
                              cudaCtxVector[dstGpuIdx],
                              dstGpuId,
                              _id2Rsrc,
                              gpuCnt,
                              width,
                              height,
                              sizeof(int));
  }

// if (doDepthTest) {
//   depthRbo = itr->second.framebufferPtr->getRenderbuffer(FboRenderBuffer::DEPTH_BUFFER);
// }

#else
  CHECK(false) << "Cuda is not enabled. It is required as EGL-based compositing relies on CUDA to do cross-gpu copy "
                  "of texture memory";
#endif
}

void EglQueryRenderCompositorImpl::_compositePass(const std::set<GpuId>& usedGpus,
                                                  size_t width,
                                                  size_t height,
                                                  bool doHitTest,
                                                  bool doDepthTest,
                                                  int passCnt,
                                                  ScaleShPtr& accumulatorScalePtr) {
#ifdef HAVE_CUDA
  _rendererPtr->makeActiveOnCurrentThread();
  _rndrCudaStateGuard->unmapAllCudaRsrcs();

  _framebufferPtr->setHitTest(accumulatorScalePtr ? false : doHitTest);
  _framebufferPtr->setDepthTest(doDepthTest);
  _framebufferPtr->bindToRenderer(_rendererPtr.get());

  _rendererPtr->setViewport(0, 0, width, height);
  if (passCnt == 0) {
    _rendererPtr->setClearColor(0, 0, 0, 0);
    _rendererPtr->clearAll();
  }

  if (accumulatorScalePtr) {
    // NOTE: the blend equations should have been set in QueryRenderer
    _rendererPtr->enable(GL_BLEND);
    _rendererPtr->setBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    _rendererPtr->setBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    accumulatorScalePtr->renderAccumulation(
        _rendererPtr.get(), _rendererPtr->getGpuId(), _accumulationTextureArray.get());

    if (!doHitTest) {
      // can just return at this point
      // TODO(croot): this may need to change at some point if
      // other things need to be rendered with an accumulation
      // Depth? Alternate rendering styles (i.e. stroking?)
      return;
    }

    _framebufferPtr->setHitTest(doHitTest);
    _framebufferPtr->setDepthTest(doDepthTest);
    _framebufferPtr->bindToRenderer(_rendererPtr.get());
  }

  _rendererPtr->bindShader(_shader);
  _rendererPtr->bindVertexArray(_vao);

  // we're doing the blending manually in the shader, so disable any blending here
  // TODO(croot): but what about multiple passes? Is this going to work then?
  // I think this will be a bug!!!!
  // _rendererPtr->disable(GL_BLEND);

  // When multi-sampling, we want to read from all the samples too,
  // so enable sample shading to enforce that.
  _rendererPtr->enable(GL_SAMPLE_SHADING);
  _rendererPtr->setMinSampleShading(1.0);

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
  _rendererPtr->drawVertexBuffers(GL_TRIANGLE_STRIP);

  // TODO(croot): push/pop a state?
  _rendererPtr->disable(GL_SAMPLE_SHADING);
#endif
}

void EglQueryRenderCompositorImpl::render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) {
#ifdef HAVE_CUDA
  QueryRendererContext* ctx = queryRenderer->getContext();
  auto qrmPerGpuDataPtr = ctx->getRootGpuCache()->perGpuData;

  bool doHitTest = ctx->doHitTest();
  // bool doDepthTest = ctx->doDepthTest();

  if (_rebuildCudaConsumedRsrcs) {
    auto& cudaCtxVector = _cudaMgr->getDeviceContexts();
    for (auto& itr : _consumedRsrcs) {
      auto perGpuItem = qrmPerGpuDataPtr->find(itr.first);
      CHECK(perGpuItem != qrmPerGpuDataPtr->end());
      (*perGpuItem)->makeActiveOnCurrentThread();
      auto gpuIdx = itr.first - _startGpuId;
      CudaStateGuard cudaStateGuard(cudaCtxVector[gpuIdx], itr.first);
      for (auto rsrcItr = itr.second.rgbaTextures.begin(); rsrcItr != itr.second.rgbaTextures.end(); rsrcItr++) {
        CHECK(!rsrcItr->txWkPtr.expired());
        itr.second.rgbaTextures.modify(rsrcItr,
                                       UpdateCudaRsrc(initCudaGraphicsResource(itr.first,
                                                                               rsrcItr->txPtr->getId(),
                                                                               rsrcItr->txPtr->getTarget(),
                                                                               CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)));
      }

      for (auto rsrcItr = itr.second.id1ATextures.begin(); rsrcItr != itr.second.id1ATextures.end(); rsrcItr++) {
        CHECK(!rsrcItr->txWkPtr.expired());
        itr.second.id1ATextures.modify(rsrcItr,
                                       UpdateCudaRsrc(initCudaGraphicsResource(itr.first,
                                                                               rsrcItr->txPtr->getId(),
                                                                               rsrcItr->txPtr->getTarget(),
                                                                               CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)));
      }

      for (auto rsrcItr = itr.second.id1BTextures.begin(); rsrcItr != itr.second.id1BTextures.end(); rsrcItr++) {
        CHECK(!rsrcItr->txWkPtr.expired());
        itr.second.id1BTextures.modify(rsrcItr,
                                       UpdateCudaRsrc(initCudaGraphicsResource(itr.first,
                                                                               rsrcItr->txPtr->getId(),
                                                                               rsrcItr->txPtr->getTarget(),
                                                                               CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)));
      }

      for (auto rsrcItr = itr.second.id2Textures.begin(); rsrcItr != itr.second.id2Textures.end(); rsrcItr++) {
        CHECK(!rsrcItr->txWkPtr.expired());
        itr.second.id2Textures.modify(rsrcItr,
                                      UpdateCudaRsrc(initCudaGraphicsResource(itr.first,
                                                                              rsrcItr->txPtr->getId(),
                                                                              rsrcItr->txPtr->getTarget(),
                                                                              CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)));
      }

      // TODO(croot): handle depth
    }
    _rebuildCudaConsumedRsrcs = false;
  }

  CudaStateGuard cudaCtxGuard(this);
  _rndrCudaStateGuard = &cudaCtxGuard;

  GLTexture2dShPtr rgbaTex, idTex, id2Tex;
  GLRenderbufferShPtr depthRbo;

  size_t numFbos = 0;
  for (auto& itr : _consumedRsrcs) {
    auto rgbaSz = itr.second.rgbaTextures.size();
    numFbos += rgbaSz;
    if (doHitTest) {
      CHECK(itr.second.id1ATextures.size() == rgbaSz);
      if (_id1BTextureArray) {
        CHECK(itr.second.id1BTextures.size() == rgbaSz);
      }
      CHECK(itr.second.id2Textures.size() == rgbaSz);
    }
  }
  auto gpuId = _rendererPtr->getGpuId();
  CHECK(_rgbaTextureArray && numFbos);
  if (_rgbaTextureArray->getDepth() != numFbos) {
    _rendererPtr->makeActiveOnCurrentThread();
    destroyCudaGraphicsResource(gpuId, _rgbaRsrc);
    _rgbaTextureArray->resize(numFbos);
    _rgbaRsrc = initCudaGraphicsResource(
        gpuId, _rgbaTextureArray->getId(), _rgbaTextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    CHECK(numFbos == _rgbaTextureArray->getDepth());
  }

  if (doHitTest) {
    CHECK(_id1ATextureArray && _id2TextureArray);
    if (_id1ATextureArray->getDepth() != numFbos) {
      _rendererPtr->makeActiveOnCurrentThread();
      destroyCudaGraphicsResource(gpuId, _id1ARsrc);
      _id1ATextureArray->resize(numFbos);
      _id1ARsrc = initCudaGraphicsResource(
          gpuId, _id1ATextureArray->getId(), _id1ATextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
      CHECK(numFbos == _id1ATextureArray->getDepth());
    }

    if (_id1BTextureArray && _id1BTextureArray->getDepth() != numFbos) {
      _rendererPtr->makeActiveOnCurrentThread();
      destroyCudaGraphicsResource(gpuId, _id1BRsrc);
      _id1BTextureArray->resize(numFbos);
      _id1BRsrc = initCudaGraphicsResource(
          gpuId, _id1BTextureArray->getId(), _id1BTextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
      CHECK(numFbos == _id1BTextureArray->getDepth());
    }

    if (_id2TextureArray->getDepth() != numFbos) {
      _rendererPtr->makeActiveOnCurrentThread();
      destroyCudaGraphicsResource(gpuId, _id2Rsrc);
      _id2TextureArray->resize(numFbos);
      _id2Rsrc = initCudaGraphicsResource(
          gpuId, _id2TextureArray->getId(), _id2TextureArray->getTarget(), CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
      CHECK(numFbos == _id2TextureArray->getDepth());
    }
  }
  // if (doDepthTest) {
  //   CHECK(_rbos.size() == _)
  // }

  QueryRenderer::renderPasses(qrmPerGpuDataPtr,
                              ctx,
                              usedGpus,
                              true,
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
#endif
}

}  // namespace EGL
}  // namespace Impl
}  // namespace QueryRenderer
