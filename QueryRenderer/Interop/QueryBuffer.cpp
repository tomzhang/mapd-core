#include "QueryBuffer.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/Resources/GLBaseBuffer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>

namespace QueryRenderer {

QueryBuffer::QueryBuffer(BufType type) : _type(type), _isActive(false), _usedBytes(0) {
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

      checkCudaErrors(result);
      for (const auto& item : _cudaResourceMap) {
        checkCudaErrors(cuCtxSetCurrent(item.first));
        checkCudaErrors(cuGraphicsUnregisterResource(item.second));
      }
      checkCudaErrors(cuCtxSetCurrent(currCudaCtx));
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
CudaHandle QueryBuffer::getCudaHandlePreQuery() {
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
  // checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, _cudaResource));
  checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, cudaRsrc));

  LOG_IF(WARNING, num_bytes != numBufBytes) << "QueryBuffer " << gpuId << ": couldn't successfully map all "
                                            << numBufBytes << " bytes. Was only able to map " << num_bytes
                                            << " bytes for cuda.";

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

#ifdef HAVE_CUDA
// static void checkCudaErrors(CUresult result);
void QueryBuffer::checkCudaErrors(CUresult result) {
  if (result == CUDA_ERROR_INVALID_GRAPHICS_CONTEXT) {
    ::Rendering::GL::GLRenderer* renderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    if (!renderer) {
      CHECK(false) << "CUDA error code=" << result << ". No gl context is set as current.";
    } else {
      CHECK(false) << "CUDA error code=" << result << ". Current context is on gpu: " << renderer->getGpuId()
                   << ", but expecting it to be on " << _getGpuId() << ".";
    }
  } else {
    CHECK(result == CUDA_SUCCESS) << "CUDA error code=" << result;
  }
}

CUgraphicsResource QueryBuffer::_getCudaGraphicsResource(bool registerResource) {
  CUcontext currCudaCtx;
  CUdevice ctxDevice;

  checkCudaErrors(cuCtxGetCurrent(&currCudaCtx));
  checkCudaErrors(cuCtxGetDevice(&ctxDevice));

  size_t gpuId = _getGpuId();
  RUNTIME_EX_ASSERT(ctxDevice == static_cast<int>(gpuId),
                    "QueryResultVertexBuffer " + std::to_string(gpuId) +
                        ": Invalid cuda context for QueryResultVertexBuffer. Device " + std::to_string(ctxDevice) +
                        " for cuda context does not match the QueryResultVertexBuffer device " + std::to_string(gpuId));

  // TODO(croot): convert these checks to log/throw errors
  CHECK(currCudaCtx != nullptr);
  CHECK(_bufRsrc);
  GLuint rsrcId = _bufRsrc->getId();
  CHECK(rsrcId);

  const auto itr = _cudaResourceMap.find(currCudaCtx);
  if (itr == _cudaResourceMap.end()) {
    CHECK(registerResource);

    CUgraphicsResource rsrc;
    checkCudaErrors(cuGraphicsGLRegisterBuffer(&rsrc, rsrcId, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
    _cudaResourceMap.insert(std::make_pair(currCudaCtx, rsrc));
    return rsrc;
  }

  return itr->second;
}

void QueryBuffer::_mapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  if (_mappedCudaResources.find(rsrc) == _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsMapResources(1, &rsrc, 0));
    _mappedCudaResources.insert(rsrc);
  }  // already mapped otherwise
}

void QueryBuffer::_unmapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  const auto itr = _mappedCudaResources.find(rsrc);
  if (itr != _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsUnmapResources(1, &rsrc, 0));
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
}

QueryResultIndirectIbo::~QueryResultIndirectIbo() {
}

}  // namespace QueryRenderer
