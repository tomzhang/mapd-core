#include "QueryResultVertexBuffer.h"
#include <Rendering/Renderer/GL/GLResourceManager.h>

namespace QueryRenderer {

using ::Rendering::GL::Resources::BufferAccessType;
using ::Rendering::GL::Resources::BufferAccessFreq;

QueryVertexBuffer::QueryVertexBuffer(VboType type) : _type(type) {
}

// QueryVertexBuffer::QueryVertexBuffer(Rendering::GL::GLRenderer* renderer, BufferAccessType accessType,
// BufferAccessFreq accessFreq) : _vbo(nullptr) {
// }

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
      _gpuId(-1),
      _cudaResourceMap() {
  _gpuId = renderer->getGpuId();
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

void QueryResultVertexBuffer::updatePostQuery(size_t numUsedBytes) {
  // TODO(croot): fill this function in. Should be called after the query is completed
  // and this buffer is filled with data. We just need to know what's in that data.

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
void QueryResultVertexBuffer::checkCudaErrors(CUresult result) {
  if (result == CUDA_ERROR_INVALID_GRAPHICS_CONTEXT) {
    ::Rendering::GL::GLRenderer* renderer = ::Rendering::GL::GLRenderer::getCurrentThreadRenderer();
    if (!renderer) {
      CHECK(false) << "CUDA error code=" << result << ". No gl context is set as current.";
    } else {
      CHECK(false) << "CUDA error code=" << result << ". Current context is on gpu: " << renderer->getGpuId()
                   << ", but expecting it to be on " << _gpuId << ".";
    }
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
  CUgraphicsResource cudaRsrc = _getCudaGraphicsResource(true);
  _mapCudaGraphicsResource(cudaRsrc);

  size_t num_bytes = 0;
  size_t numVboBytes = _vbo->numBytes();

  CUdeviceptr devPtr;
  // checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, _cudaResource));
  checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, cudaRsrc));

  LOG_IF(WARNING, num_bytes != numVboBytes) << "QueryResultVertexBuffer " << _gpuId
                                            << ": couldn't successfully map all " << numVboBytes
                                            << " bytes. Was only able to map " << num_bytes << " bytes for cuda.";

  _isActive = true;

  return CudaHandle(reinterpret_cast<void*>(devPtr), num_bytes);
}

CUgraphicsResource QueryResultVertexBuffer::_getCudaGraphicsResource(bool registerResource) {
  // CUcontext* currCudaCtx=nullptr;
  CUcontext currCudaCtx;
  CUdevice ctxDevice;

  checkCudaErrors(cuCtxGetCurrent(&currCudaCtx));
  checkCudaErrors(cuCtxGetDevice(&ctxDevice));

  RUNTIME_EX_ASSERT(ctxDevice == _gpuId,
                    "QueryResultVertexBuffer " + std::to_string(_gpuId) +
                        ": Invalid cuda context for QueryResultVertexBuffer. Device " + std::to_string(ctxDevice) +
                        " for cuda context does not match the QueryResultVertexBuffer device " +
                        std::to_string(_gpuId));

  // TODO(croot): convert these checks to log/throw errors
  CHECK(currCudaCtx != nullptr);
  CHECK(_vbo);
  GLuint rsrcId = _vbo->getId();
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

void QueryResultVertexBuffer::_mapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  if (_mappedCudaResources.find(rsrc) == _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsMapResources(1, &rsrc, 0));
    _mappedCudaResources.insert(rsrc);
  }  // already mapped otherwise
}

void QueryResultVertexBuffer::_unmapCudaGraphicsResource(CUgraphicsResource& rsrc) {
  const auto itr = _mappedCudaResources.find(rsrc);
  if (itr != _mappedCudaResources.end()) {
    checkCudaErrors(cuGraphicsUnmapResources(1, &rsrc, 0));
    _mappedCudaResources.erase(itr);
  }
  // TODO(croot): throw an error if nothing to unmap?
}
#endif  // HAVE_CUDA

}  // namespace QueryRenderer
