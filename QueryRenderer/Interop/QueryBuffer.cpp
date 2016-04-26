#include "QueryBuffer.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/Resources/GLBaseBuffer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>

namespace QueryRenderer {

QueryBuffer::QueryBuffer(BufType type)
    : _type(type),
      _isActive(false),
#ifdef HAVE_CUDA
      _usedBytes(0),
      _cudaResourceMap() {
#else
      _usedBytes(0) {
#endif  // HAVE_CUDA
}

QueryBuffer::~QueryBuffer() {
#ifdef HAVE_CUDA
  // NOTE:   there can only be one registered graphics resource per context
  // see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graphics-interoperability
  // as it states in that document:
  // "Each CUDA context which intends to use the resource is required to register it separately."
  // So, we are maintaining registered resources via an unordered_map by context.
  // TODO(croot): should we let the caller be responsible for registering the resource
  // instead and pass it as an argument?

  // return to the previous cuda context after
  // unregistering all remaining resources
  // for each context
  if (_cudaResourceMap.size()) {
    CUcontext currCudaCtx;
    CUresult result = cuCtxGetCurrent(&currCudaCtx);

    // NOTE: We're not checking for CUDA_ERROR_NOT_INITIALIZED
    // since CUDA should have been initialized if there's something in
    // _cudaResourceMap.
    if (result != CUDA_ERROR_DEINITIALIZED) {
      // verify that cuda's not being shut down.
      // This destructor may have been triggered
      // by a program exit, therefore cuda could
      // be in shutdown mode before we get here.

      checkCudaErrors(result, __FILE__, __LINE__);
      for (const auto& item : _cudaResourceMap) {
        checkCudaErrors(cuCtxSetCurrent(item.first), __FILE__, __LINE__);
        checkCudaErrors(cuGraphicsUnregisterResource(item.second), __FILE__, __LINE__);
      }
      checkCudaErrors(cuCtxSetCurrent(currCudaCtx), __FILE__, __LINE__);
    }
  }
#endif  // HAVE_CUDA
}

size_t QueryBuffer::_getGpuId() const {
  CHECK(_bufRsrc != nullptr);
  ::Rendering::GL::GLRenderer* renderer = _bufRsrc->getGLRenderer();
  CHECK(renderer);
  return renderer->getGpuId();
}

#ifdef HAVE_CUDA
CudaHandle QueryBuffer::getCudaHandlePreQuery(bool useAllAllocatedBytes) {
  // Handling the state of the buffer since the GL VBO needs to be mapped/unmapped to/from a CUDA buffer.
  // Managing the state ensures that the mapping/unmapping is done in the appropriate order.

  // TODO(croot): make thread safe? There is already a mutex in the QueryRenderManager class
  // so it is essentially thread safe now, but perhaps that mutex should be
  // here, or an additional one here. Might improve thread performance.

  CHECK(_type == BufType::QUERY_RESULT_BUFFER);
  size_t gpuId = _getGpuId();

  RUNTIME_EX_ASSERT(!_isActive,
                    "QueryBuffer " + std::to_string(gpuId) + " is already in use. Cannot access cuda handle.");

  // now map the buffer for cuda
  CUgraphicsResource cudaRsrc = _getCudaGraphicsResource(true);
  _mapCudaGraphicsResource(cudaRsrc);

  size_t num_bytes = 0;
  size_t numBufBytes = _bufRsrc->numBytes();

  CUdeviceptr devPtr;
  checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, cudaRsrc), __FILE__, __LINE__);

  if (useAllAllocatedBytes) {
    LOG_IF(ERROR, num_bytes != numBufBytes) << "QueryBuffer " << gpuId << ": couldn't successfully map all "
                                            << numBufBytes << " bytes. Was only able to map " << num_bytes
                                            << " bytes for cuda.";

  } else {
    LOG_IF(WARNING, num_bytes != numBufBytes) << "QueryBuffer " << gpuId << ": couldn't successfully map all "
                                              << numBufBytes << " bytes. Was only able to map " << num_bytes
                                              << " bytes for cuda.";
  }

  _isActive = true;

  return CudaHandle(reinterpret_cast<void*>(devPtr), num_bytes);
}
#endif

void QueryBuffer::updatePostQuery(size_t numUsedBytes) {
  CHECK(_type == BufType::QUERY_RESULT_BUFFER);

  RUNTIME_EX_ASSERT(
      _isActive,
      "QueryBuffer " + std::to_string(_getGpuId()) + " has not been prepped for a query. Cannot set data post query.");

  _isActive = false;

#ifdef HAVE_CUDA
  // unmap buffer object
  CUgraphicsResource cudaRsrc = _getCudaGraphicsResource();
  _unmapCudaGraphicsResource(cudaRsrc);
#endif

  _usedBytes = numUsedBytes;
}

void QueryBuffer::resize(size_t numBytes) {
  RUNTIME_EX_ASSERT(!_isActive, "Cannot resize a buffer while it is currently mapped for cuda.");

  if (_bufRsrc) {
    if (numBytes != _bufRsrc->numBytes()) {
      _bufRsrc->bufferData(nullptr, numBytes);

#ifdef HAVE_CUDA
      if (_type == BufType::QUERY_RESULT_BUFFER) {
        _removeCudaGraphicsResource();
        _initCudaGraphicsResource();
      }
#endif  // HAVE_CUDA
    }

    reset();
  }
}

#ifdef HAVE_CUDA
void QueryBuffer::checkCudaErrors(CUresult result, const char* filename, int lineno) {
  if (result == CUDA_ERROR_INVALID_GRAPHICS_CONTEXT) {
    ::Rendering::GL::GLRenderer* renderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    if (!renderer) {
      CHECK(false) << "CUDA error code=" << result << ". No gl context is set as current. " << filename << ":"
                   << lineno;
    } else {
      CHECK(false) << "CUDA error code=" << result << ". Current context is on gpu: " << renderer->getGpuId()
                   << ", but expecting it to be on " << _getGpuId() << ". " << filename << ":" << lineno;
    }
  } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    auto gpuId = _getGpuId();
    LOG(ERROR) << "CUDA out of memory error on gpu: " << gpuId << " during rendering. " << filename << ":" << lineno;
    throw Rendering::OutOfGpuMemoryError("Cuda out of memory error on gpu: " + std::to_string(gpuId) +
                                         " - CUDA error code =" + std::to_string(result));
  } else {
    CHECK(result == CUDA_SUCCESS) << "CUDA error code=" << result;
  }
}

void QueryBuffer::_initCudaGraphicsResource() {
  CUcontext currCudaCtx;
  CUdevice ctxDevice;

  CHECK(_type == BufType::QUERY_RESULT_BUFFER);

  checkCudaErrors(cuCtxGetCurrent(&currCudaCtx), __FILE__, __LINE__);
  checkCudaErrors(cuCtxGetDevice(&ctxDevice), __FILE__, __LINE__);

  auto gpuId = _getGpuId();
  RUNTIME_EX_ASSERT(ctxDevice == static_cast<int>(gpuId),
                    "QueryBuffer " + std::to_string(gpuId) + ": Invalid cuda context for QueryBuffer. Device " +
                        std::to_string(ctxDevice) + " for cuda context does not match the QueryBuffer device " +
                        std::to_string(gpuId));

  CHECK(currCudaCtx != nullptr);
  CHECK(_bufRsrc);
  GLuint rsrcId = _bufRsrc->getId();
  CHECK(rsrcId);

  const auto itr = _cudaResourceMap.find(currCudaCtx);
  if (itr == _cudaResourceMap.end()) {
    CUgraphicsResource rsrc;
    CUresult result = cuGraphicsGLRegisterBuffer(&rsrc, rsrcId, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

    if (result == CUDA_ERROR_UNKNOWN) {
      throw ::Rendering::OutOfGpuMemoryError(
          "Cuda error code=" + std::to_string(result) +
          ", CUDA_ERROR_UNKNOWN, possibly not enough gpu memory available for the requested buffer size of " +
          std::to_string(_bufRsrc->numBytes()) + " bytes.");
    }

    checkCudaErrors(result, __FILE__, __LINE__);
    _cudaResourceMap.insert(std::make_pair(currCudaCtx, rsrc));
  }
}

void QueryBuffer::_removeCudaGraphicsResource() {
  CUcontext currCudaCtx;
  CUdevice ctxDevice;

  CHECK(_type == BufType::QUERY_RESULT_BUFFER);

  checkCudaErrors(cuCtxGetCurrent(&currCudaCtx), __FILE__, __LINE__);
  checkCudaErrors(cuCtxGetDevice(&ctxDevice), __FILE__, __LINE__);

  auto gpuId = _getGpuId();
  RUNTIME_EX_ASSERT(ctxDevice == static_cast<int>(gpuId),
                    "QueryBuffer " + std::to_string(gpuId) + ": Invalid cuda context for QueryBuffer. Device " +
                        std::to_string(ctxDevice) + " for cuda context does not match the QueryBuffer device " +
                        std::to_string(gpuId));

  CHECK(currCudaCtx != nullptr);
  CHECK(_bufRsrc);
  GLuint rsrcId = _bufRsrc->getId();
  CHECK(rsrcId);

  const auto itr = _cudaResourceMap.find(currCudaCtx);
  if (itr != _cudaResourceMap.end()) {
    checkCudaErrors(cuGraphicsUnregisterResource(itr->second), __FILE__, __LINE__);

    _cudaResourceMap.erase(itr);
  }
}

CUgraphicsResource QueryBuffer::_getCudaGraphicsResource(bool registerResource) {
  CUcontext currCudaCtx;
  CUdevice ctxDevice;

  CHECK(_type == BufType::QUERY_RESULT_BUFFER);

  checkCudaErrors(cuCtxGetCurrent(&currCudaCtx), __FILE__, __LINE__);
  checkCudaErrors(cuCtxGetDevice(&ctxDevice), __FILE__, __LINE__);

  auto gpuId = _getGpuId();
  RUNTIME_EX_ASSERT(ctxDevice == static_cast<int>(gpuId),
                    "QueryBuffer " + std::to_string(gpuId) + ": Invalid cuda context for QueryBuffer. Device " +
                        std::to_string(ctxDevice) + " for cuda context does not match the QueryBuffer device " +
                        std::to_string(gpuId));

  CHECK(currCudaCtx != nullptr);
  CHECK(_bufRsrc);
  GLuint rsrcId = _bufRsrc->getId();
  CHECK(rsrcId);

  const auto itr = _cudaResourceMap.find(currCudaCtx);
  CHECK(itr != _cudaResourceMap.end());
  return itr->second;
}

void QueryBuffer::_mapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  if (_mappedCudaResources.find(rsrc) == _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsMapResources(1, &rsrc, 0), __FILE__, __LINE__);
    _mappedCudaResources.insert(rsrc);
  }  // already mapped otherwise
}

void QueryBuffer::_unmapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  const auto itr = _mappedCudaResources.find(rsrc);
  if (itr != _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsUnmapResources(1, &rsrc, 0), __FILE__, __LINE__);
    _mappedCudaResources.erase(itr);
  }
  // TODO(croot): throw an error if nothing to unmap?
}

#endif  // HAVE_CUDA

QueryVertexBuffer::QueryVertexBuffer(QueryBuffer::BufType type)
    : QueryLayoutBuffer<::Rendering::GL::Resources::GLVertexBuffer>(type) {
}

QueryVertexBuffer::QueryVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                     size_t numBytes,
                                     Rendering::GL::Resources::BufferAccessType accessType,
                                     Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                     QueryBuffer::BufType type)
    : QueryLayoutBuffer<::Rendering::GL::Resources::GLVertexBuffer>(type) {
  _initBuffer(renderer, accessType, accessFreq, numBytes);
}

QueryVertexBuffer::QueryVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                     const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr,
                                     Rendering::GL::Resources::BufferAccessType accessType,
                                     Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                     QueryBuffer::BufType type)
    : QueryLayoutBuffer<::Rendering::GL::Resources::GLVertexBuffer>(type) {
  _initLayoutBuffer(renderer, accessType, accessFreq, layoutPtr);
}

QueryVertexBuffer::~QueryVertexBuffer() {
}

void QueryVertexBuffer::_initBuffer(Rendering::GL::GLRenderer* renderer,
                                    Rendering::GL::Resources::BufferAccessType accessType,
                                    Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                    size_t numBytes) {
  if (!_bufRsrc) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

    if (numBytes > 0) {
      _bufRsrc = rsrcMgr->createVertexBuffer(numBytes, accessType, accessFreq);
    } else {
      _bufRsrc = rsrcMgr->createVertexBuffer(accessType, accessFreq);
    }
  }
}

void QueryVertexBuffer::_initLayoutBuffer(Rendering::GL::GLRenderer* renderer,
                                          Rendering::GL::Resources::BufferAccessType accessType,
                                          Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                          const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr) {
  if (!_bufRsrc) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

    _bufRsrc = rsrcMgr->createVertexBuffer(layoutPtr, accessType, accessFreq);
  }
}

QueryResultVertexBuffer::QueryResultVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                                 size_t numBytes,
                                                 Rendering::GL::Resources::BufferAccessType accessType,
                                                 Rendering::GL::Resources::BufferAccessFreq accessFreq)
    : QueryVertexBuffer(renderer, numBytes, accessType, accessFreq, QueryBuffer::BufType::QUERY_RESULT_BUFFER) {
#ifdef HAVE_CUDA
  _initCudaGraphicsResource();
#endif
}

QueryResultVertexBuffer::~QueryResultVertexBuffer() {
}

QueryIndexBuffer::QueryIndexBuffer(QueryBuffer::BufType type) : QueryBuffer(type) {
}

QueryIndexBuffer::QueryIndexBuffer(Rendering::GL::GLRenderer* renderer,
                                   size_t numBytes,
                                   ::Rendering::GL::Resources::GLIndexBuffer::IndexType indexType,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  _initBuffer(renderer, indexType, accessType, accessFreq, numBytes);
}

QueryIndexBuffer::QueryIndexBuffer(::Rendering::GL::GLRenderer* renderer,
                                   const std::vector<unsigned char>& items,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
  _bufRsrc = rsrcMgr->createIndexBuffer(items, accessType, accessFreq);
}

QueryIndexBuffer::QueryIndexBuffer(::Rendering::GL::GLRenderer* renderer,
                                   const std::vector<unsigned short>& items,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
  _bufRsrc = rsrcMgr->createIndexBuffer(items, accessType, accessFreq);
}

QueryIndexBuffer::QueryIndexBuffer(::Rendering::GL::GLRenderer* renderer,
                                   const std::vector<unsigned int>& items,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
  _bufRsrc = rsrcMgr->createIndexBuffer(items, accessType, accessFreq);
}

QueryIndexBuffer::~QueryIndexBuffer() {
}

void QueryIndexBuffer::_initBuffer(::Rendering::GL::GLRenderer* renderer,
                                   ::Rendering::GL::Resources::GLIndexBuffer::IndexType indexType,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   size_t numBytes) {
  if (!_bufRsrc) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

    if (numBytes > 0) {
      _bufRsrc = rsrcMgr->createIndexBuffer(numBytes, indexType, accessType, accessFreq);
    } else {
      _bufRsrc = rsrcMgr->createIndexBuffer(indexType, accessType, accessFreq);
    }
  }
}

QueryResultIndexBuffer::QueryResultIndexBuffer(Rendering::GL::GLRenderer* renderer,
                                               size_t numBytes,
                                               ::Rendering::GL::Resources::GLIndexBuffer::IndexType indexType,
                                               ::Rendering::GL::Resources::BufferAccessType accessType,
                                               ::Rendering::GL::Resources::BufferAccessFreq accessFreq)
    : QueryIndexBuffer(renderer,
                       numBytes,
                       indexType,
                       accessType,
                       accessFreq,
                       QueryBuffer::BufType::QUERY_RESULT_BUFFER) {
#ifdef HAVE_CUDA
  _initCudaGraphicsResource();
#endif
}

QueryResultIndexBuffer::~QueryResultIndexBuffer() {
}

QueryUniformBuffer::QueryUniformBuffer(QueryBuffer::BufType type)
    : QueryLayoutBuffer<::Rendering::GL::Resources::GLUniformBuffer, ::Rendering::GL::Resources::GLShaderBlockLayout>(
          type) {
}

QueryUniformBuffer::QueryUniformBuffer(Rendering::GL::GLRenderer* renderer,
                                       size_t numBytes,
                                       ::Rendering::GL::Resources::BufferAccessType accessType,
                                       ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                       QueryBuffer::BufType type)
    : QueryLayoutBuffer<::Rendering::GL::Resources::GLUniformBuffer, ::Rendering::GL::Resources::GLShaderBlockLayout>(
          type) {
  _initBuffer(renderer, accessType, accessFreq, numBytes);
}

QueryUniformBuffer::QueryUniformBuffer(::Rendering::GL::GLRenderer* renderer,
                                       const ::Rendering::GL::Resources::GLShaderBlockLayoutShPtr& layoutPtr,
                                       ::Rendering::GL::Resources::BufferAccessType accessType,
                                       ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                       QueryBuffer::BufType type)
    : QueryLayoutBuffer<::Rendering::GL::Resources::GLUniformBuffer, ::Rendering::GL::Resources::GLShaderBlockLayout>(
          type) {
  _initLayoutBuffer(renderer, accessType, accessFreq, layoutPtr);
}

QueryUniformBuffer::~QueryUniformBuffer() {
}

void QueryUniformBuffer::_initBuffer(::Rendering::GL::GLRenderer* renderer,
                                     ::Rendering::GL::Resources::BufferAccessType accessType,
                                     ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                     size_t numBytes) {
  if (!_bufRsrc) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

    if (numBytes > 0) {
      _bufRsrc = rsrcMgr->createUniformBuffer(numBytes, accessType, accessFreq);
    } else {
      _bufRsrc = rsrcMgr->createUniformBuffer(accessType, accessFreq);
    }
  }
}

void QueryUniformBuffer::_initLayoutBuffer(Rendering::GL::GLRenderer* renderer,
                                           Rendering::GL::Resources::BufferAccessType accessType,
                                           Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                           const ::Rendering::GL::Resources::GLShaderBlockLayoutShPtr& layoutPtr) {
  if (!_bufRsrc) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
    _bufRsrc = rsrcMgr->createUniformBuffer(layoutPtr, 0, accessType, accessFreq);
  }
}

QueryResultUniformBuffer::QueryResultUniformBuffer(Rendering::GL::GLRenderer* renderer,
                                                   size_t numBytes,
                                                   Rendering::GL::Resources::BufferAccessType accessType,
                                                   Rendering::GL::Resources::BufferAccessFreq accessFreq)
    : QueryUniformBuffer(renderer, numBytes, accessType, accessFreq, QueryBuffer::BufType::QUERY_RESULT_BUFFER) {
#ifdef HAVE_CUDA
  _initCudaGraphicsResource();
#endif
}

QueryResultUniformBuffer::~QueryResultUniformBuffer() {
}

QueryIndirectVbo::QueryIndirectVbo(QueryBuffer::BufType type) : QueryBuffer(type) {
}

QueryIndirectVbo::QueryIndirectVbo(Rendering::GL::GLRenderer* renderer,
                                   size_t numBytes,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  _initBuffer(renderer, accessType, accessFreq, numBytes);
}

QueryIndirectVbo::QueryIndirectVbo(::Rendering::GL::GLRenderer* renderer,
                                   const std::vector<::Rendering::GL::Resources::IndirectDrawVertexData>& items,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
  _bufRsrc = rsrcMgr->createIndirectDrawVertexBuffer(items, accessType, accessFreq);
}

QueryIndirectVbo::~QueryIndirectVbo() {
}

void QueryIndirectVbo::_initBuffer(::Rendering::GL::GLRenderer* renderer,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   size_t numBytes) {
  if (!_bufRsrc) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

    if (numBytes > 0) {
      _bufRsrc = rsrcMgr->createIndirectDrawVertexBuffer(numBytes, accessType, accessFreq);
    } else {
      _bufRsrc = rsrcMgr->createIndirectDrawVertexBuffer(accessType, accessFreq);
    }
  }
}

QueryResultIndirectVbo::QueryResultIndirectVbo(Rendering::GL::GLRenderer* renderer,
                                               size_t numBytes,
                                               ::Rendering::GL::Resources::BufferAccessType accessType,
                                               ::Rendering::GL::Resources::BufferAccessFreq accessFreq)
    : QueryIndirectVbo(renderer, numBytes, accessType, accessFreq, QueryBuffer::BufType::QUERY_RESULT_BUFFER) {
#ifdef HAVE_CUDA
  _initCudaGraphicsResource();
#endif
}

QueryResultIndirectVbo::~QueryResultIndirectVbo() {
}

QueryIndirectIbo::QueryIndirectIbo(QueryBuffer::BufType type) : QueryBuffer(type) {
}

QueryIndirectIbo::QueryIndirectIbo(Rendering::GL::GLRenderer* renderer,
                                   size_t numBytes,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  _initBuffer(renderer, accessType, accessFreq, numBytes);
}

QueryIndirectIbo::QueryIndirectIbo(::Rendering::GL::GLRenderer* renderer,
                                   const std::vector<::Rendering::GL::Resources::IndirectDrawIndexData>& items,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   QueryBuffer::BufType type)
    : QueryBuffer(type) {
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();
  _bufRsrc = rsrcMgr->createIndirectDrawIndexBuffer(items, accessType, accessFreq);
}

QueryIndirectIbo::~QueryIndirectIbo() {
}

void QueryIndirectIbo::_initBuffer(::Rendering::GL::GLRenderer* renderer,
                                   ::Rendering::GL::Resources::BufferAccessType accessType,
                                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                   size_t numBytes) {
  if (!_bufRsrc) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

    if (numBytes > 0) {
      _bufRsrc = rsrcMgr->createIndirectDrawIndexBuffer(numBytes, accessType, accessFreq);
    } else {
      _bufRsrc = rsrcMgr->createIndirectDrawIndexBuffer(accessType, accessFreq);
    }
  }
}

QueryResultIndirectIbo::QueryResultIndirectIbo(Rendering::GL::GLRenderer* renderer,
                                               size_t numBytes,
                                               ::Rendering::GL::Resources::BufferAccessType accessType,
                                               ::Rendering::GL::Resources::BufferAccessFreq accessFreq)
    : QueryIndirectIbo(renderer, numBytes, accessType, accessFreq, QueryBuffer::BufType::QUERY_RESULT_BUFFER) {
#ifdef HAVE_CUDA
  _initCudaGraphicsResource();
#endif
}

QueryResultIndirectIbo::~QueryResultIndirectIbo() {
}

}  // namespace QueryRenderer
