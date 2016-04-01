#include "QueryResultVertexBuffer.h"
#include <Rendering/Renderer/GL/GLResourceManager.h>

namespace QueryRenderer {

using ::Rendering::GL::Resources::BufferAccessType;
using ::Rendering::GL::Resources::BufferAccessFreq;

QueryVertexBuffer::QueryVertexBuffer(VboType type) : _type(type) {
}

QueryVertexBuffer::QueryVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                     size_t numBytes,
                                     BufferAccessType accessType,
                                     BufferAccessFreq accessFreq,
                                     VboType type)
    : QueryVertexBuffer(type) {
  _initBuffer(renderer, accessType, accessFreq, numBytes);
}

QueryVertexBuffer::QueryVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                     const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr,
                                     BufferAccessType accessType,
                                     BufferAccessFreq accessFreq,
                                     VboType type)
    : QueryVertexBuffer(type) {
  _initBuffer(renderer, accessType, accessFreq, 0, layoutPtr);
}

QueryVertexBuffer::~QueryVertexBuffer() {
}

void QueryVertexBuffer::_initBuffer(Rendering::GL::GLRenderer* renderer,
                                    BufferAccessType accessType,
                                    BufferAccessFreq accessFreq,
                                    size_t numBytes,
                                    const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr) {
  if (!_vbo) {
    Rendering::GL::GLResourceManagerShPtr rsrcMgr = renderer->getResourceManager();

    if (numBytes > 0) {
      _vbo = rsrcMgr->createVertexBuffer(numBytes, accessType, accessFreq);
    } else if (layoutPtr) {
      _vbo = rsrcMgr->createVertexBuffer(layoutPtr, accessType, accessFreq);
    } else {
      _vbo = rsrcMgr->createVertexBuffer(accessType, accessFreq);
    }
  }
}

bool QueryVertexBuffer::hasAttribute(const std::string& attrName) const {
  CHECK(_vbo);
  return _vbo->hasAttribute(attrName);
}

size_t QueryVertexBuffer::numItems() const {
  CHECK(_vbo);
  return _vbo->numItems();
}

::Rendering::GL::Resources::GLBufferAttrType QueryVertexBuffer::getAttributeType(const std::string& attrName) const {
  CHECK(_vbo);
  return _vbo->getAttributeType(attrName);
}

::Rendering::GL::TypeGLShPtr QueryVertexBuffer::getAttributeTypeGL(const std::string& attrName) const {
  CHECK(_vbo);
  return _vbo->getAttributeTypeGL(attrName);
}

void QueryVertexBuffer::bufferData(void* data, size_t numItems, size_t numBytesPerItem) {
  CHECK(_vbo);
  return _vbo->bufferData(data, numItems, numBytesPerItem);
}

QueryResultVertexBuffer::QueryResultVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                                 size_t numBytes,
                                                 BufferAccessType accessType,
                                                 BufferAccessFreq accessFreq)
    : QueryVertexBuffer(renderer, numBytes, accessType, accessFreq),
      _isActive(false),
      _usedBytes(0),
#ifdef HAVE_CUDA
      _gpuId(-1),
      _cudaResourceMap() {
#else
      _gpuId(-1) {
#endif
  _gpuId = renderer->getGpuId();
#ifdef HAVE_CUDA
  _initCudaGraphicsResource();
#endif
}

QueryResultVertexBuffer::~QueryResultVertexBuffer() {
#ifdef HAVE_CUDA
  // TODO(croot): this is probably a bad place to manage the cuda graphics
  // resource. Ultimately the current device/current cuda context is
  // managed by the caller, and there can only be one registered per context
  // see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graphics-interoperability
  // as it states in that document:
  // "Each CUDA context which intends to use the resource is required to register it separately."
  // So, we either need to:
  //   a) let the caller be responsible for the registering of the resource, and it be passed
  //      down as an argument,
  //   or
  //   b) we somehow have an api to access the current context, and change the
  //      resource to be an unordered_map by context.

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

void QueryResultVertexBuffer::updatePostQuery(size_t numUsedBytes) {
  RUNTIME_EX_ASSERT(_isActive,
                    "QueryResultVertexBuffer " + std::to_string(_gpuId) +
                        " has not been prepped for a query. Cannot set data post query.");

  _isActive = false;

#ifdef HAVE_CUDA
  // unmap buffer object
  CUgraphicsResource cudaRsrc = _getCudaGraphicsResource();
  _unmapCudaGraphicsResource(cudaRsrc);
#endif

  _usedBytes = numUsedBytes;
}

void QueryResultVertexBuffer::setBufferLayout(const ::Rendering::GL::Resources::GLBufferLayoutShPtr& bufferLayout) {
  size_t bytesPerVertex = bufferLayout->getNumBytesPerVertex();
  RUNTIME_EX_ASSERT(_usedBytes % bytesPerVertex == 0,
                    "QueryResultVertexBuffer " + std::to_string(_gpuId) + ": Buffer layout bytes-per-vertex " +
                        std::to_string(bytesPerVertex) +
                        " does not align with the number of used bytes in the buffer: " + std::to_string(_usedBytes) +
                        ".");
  _vbo->setBufferLayout(bufferLayout, _usedBytes / bytesPerVertex);
}

#ifdef HAVE_CUDA
void QueryResultVertexBuffer::checkCudaErrors(CUresult result, const char* filename, int lineno) {
  if (result == CUDA_ERROR_INVALID_GRAPHICS_CONTEXT) {
    ::Rendering::GL::GLRenderer* renderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    if (!renderer) {
      CHECK(false) << "CUDA error code=" << result << ". No gl context is set as current. " << filename << ":"
                   << lineno;
    } else {
      CHECK(false) << "CUDA error code=" << result << ". Current context is on gpu: " << renderer->getGpuId()
                   << ", but expecting it to be on " << _gpuId << ". " << filename << ":" << lineno;
    }
  } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    LOG(ERROR) << "CUDA out of memory error on gpu: " << _gpuId << " during rendering. " << filename << ":" << lineno;
    throw Rendering::OutOfGpuMemoryError("Cuda out of memory error on gpu: " + std::to_string(_gpuId) +
                                         " - CUDA error code =" + std::to_string(result));
  } else {
    CHECK(result == CUDA_SUCCESS) << "CUDA error code=" << result;
  }
}

CudaHandle QueryResultVertexBuffer::getCudaHandlePreQuery() {
  // Handling the state of the buffer since the GL VBO needs to be mapped/unmapped to/from a CUDA buffer.
  // Managing the state ensures that the mapping/unmapping is done in the appropriate order.

  // TODO(croot): make thread safe? There is already a mutex in the QueryRenderManager class
  // so it is essentially thread safe now, but perhaps that mutex should be
  // here, or an additional one here. Might improve thread performance.
  RUNTIME_EX_ASSERT(
      !_isActive,
      "QueryResultVertexBuffer " + std::to_string(_gpuId) + " is already in use. Cannot access cuda handle.");

  // now map the buffer for cuda
  CUgraphicsResource cudaRsrc = _getCudaGraphicsResource();
  _mapCudaGraphicsResource(cudaRsrc);

  size_t num_bytes = 0;
  size_t numVboBytes = _vbo->numBytes();

  CUdeviceptr devPtr;
  checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, cudaRsrc), __FILE__, __LINE__);

  LOG_IF(WARNING, num_bytes != numVboBytes) << "QueryResultVertexBuffer " << _gpuId
                                            << ": couldn't successfully map all " << numVboBytes
                                            << " bytes. Was only able to map " << num_bytes << " bytes for cuda.";

  _isActive = true;

  return CudaHandle(reinterpret_cast<void*>(devPtr), num_bytes);
}

void QueryResultVertexBuffer::_initCudaGraphicsResource() {
  CUcontext currCudaCtx;
  CUdevice ctxDevice;

  checkCudaErrors(cuCtxGetCurrent(&currCudaCtx), __FILE__, __LINE__);
  checkCudaErrors(cuCtxGetDevice(&ctxDevice), __FILE__, __LINE__);

  RUNTIME_EX_ASSERT(ctxDevice == _gpuId,
                    "QueryResultVertexBuffer " + std::to_string(_gpuId) +
                        ": Invalid cuda context for QueryResultVertexBuffer. Device " + std::to_string(ctxDevice) +
                        " for cuda context does not match the QueryResultVertexBuffer device " +
                        std::to_string(_gpuId));

  CHECK(currCudaCtx != nullptr);
  CHECK(_vbo);
  GLuint rsrcId = _vbo->getId();
  CHECK(rsrcId);

  const auto itr = _cudaResourceMap.find(currCudaCtx);
  if (itr == _cudaResourceMap.end()) {
    CUgraphicsResource rsrc;
    CUresult result = cuGraphicsGLRegisterBuffer(&rsrc, rsrcId, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

    if (result == CUDA_ERROR_UNKNOWN) {
      throw ::Rendering::OutOfGpuMemoryError(
          "Cuda error code=" + std::to_string(result) +
          ", CUDA_ERROR_UNKNOWN, possibly not enough gpu memory available for the requested buffer size of " +
          std::to_string(_vbo->numBytes()) + " bytes.");
    }

    checkCudaErrors(result, __FILE__, __LINE__);
    _cudaResourceMap.insert(std::make_pair(currCudaCtx, rsrc));
  }
}

CUgraphicsResource QueryResultVertexBuffer::_getCudaGraphicsResource() {
  CUcontext currCudaCtx;
  CUdevice ctxDevice;

  checkCudaErrors(cuCtxGetCurrent(&currCudaCtx), __FILE__, __LINE__);
  checkCudaErrors(cuCtxGetDevice(&ctxDevice), __FILE__, __LINE__);

  RUNTIME_EX_ASSERT(ctxDevice == _gpuId,
                    "QueryResultVertexBuffer " + std::to_string(_gpuId) +
                        ": Invalid cuda context for QueryResultVertexBuffer. Device " + std::to_string(ctxDevice) +
                        " for cuda context does not match the QueryResultVertexBuffer device " +
                        std::to_string(_gpuId));

  CHECK(currCudaCtx != nullptr);
  CHECK(_vbo);
  GLuint rsrcId = _vbo->getId();
  CHECK(rsrcId);

  const auto itr = _cudaResourceMap.find(currCudaCtx);
  CHECK(itr != _cudaResourceMap.end());
  return itr->second;
}

void QueryResultVertexBuffer::_mapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  if (_mappedCudaResources.find(rsrc) == _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsMapResources(1, &rsrc, 0), __FILE__, __LINE__);
    _mappedCudaResources.insert(rsrc);
  }  // already mapped otherwise
}

void QueryResultVertexBuffer::_unmapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  const auto itr = _mappedCudaResources.find(rsrc);
  if (itr != _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsUnmapResources(1, &rsrc, 0), __FILE__, __LINE__);
    _mappedCudaResources.erase(itr);
  }
  // TODO(croot): throw an error if nothing to unmap?
}
#endif  // HAVE_CUDA

}  // namespace QueryRenderer
