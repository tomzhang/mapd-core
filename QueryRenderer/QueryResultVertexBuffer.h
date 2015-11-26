#ifndef QUERY_RESULT_VERTEX_BUFFER_H_
#define QUERY_RESULT_VERTEX_BUFFER_H_

#include "QueryRendererError.h"
#include "BufferLayout.h"
#include "Shader.h"
#include "VertexBuffer.h"
#include <GL/glew.h>
#include <glog/logging.h>

// CUDA libs
// #include <cuda_runtime.h>
// #include <cuda_gl_interop.h>
#include <cuda.h>
#include <cudaGL.h>

#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace MapD_Renderer {

struct CudaHandle {
  void* handle;

  // GLuint handle;
  unsigned int numBytes;

  CudaHandle(void* handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}

  // CudaHandle(GLuint handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}
};

// TODO(croot): create a base VBO class that both the QueryResultVertexBuffer and VertexBuffer classes can
// inherit from
class QueryResultVertexBuffer : public BaseVertexBuffer {
 public:
  explicit QueryResultVertexBuffer(unsigned int numBytes,
                                   GLenum target = GL_ARRAY_BUFFER,
                                   GLenum usage = GL_DYNAMIC_COPY)
      : BaseVertexBuffer(BaseVertexBuffer::VertexBufferType::QUERY_RESULT_VERTEX_BUFFER, target, usage),
        _isActive(false),
        _numTotalBytes(numBytes),
        // _cudaResource(nullptr),
        _cudaResourceMap() {}

  ~QueryResultVertexBuffer() {
    if (_bufferId) {
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

      // TODO(croot)
      // this keeps erroring out and I'm not sure why
      // but comment out until I do
      // checkCudaErrors(cudaGraphicsUnregisterResource(_cudaResource));
      // checkCudaErrors(cuGraphicsUnregisterResource(_cudaResource));

      // return to the previous cuda context after
      // unregistering all remaining resources
      // for each context
      CUcontext currCudaCtx;
      CUresult result = cuCtxGetCurrent(&currCudaCtx);
      if (result != CUDA_ERROR_DEINITIALIZED) {
        // verify that cuda's not being shut down.
        // This destructor may have been triggered
        // by a program exit, therefore cuda could
        // be in shutdown mode before we get here.

        checkCudaErrors(result);
        checkCudaErrors(cuCtxGetCurrent(&currCudaCtx));
        for (const auto& item : _cudaResourceMap) {
          checkCudaErrors(cuCtxSetCurrent(item.first));
          checkCudaErrors(cuGraphicsUnregisterResource(item.second));
        }
        checkCudaErrors(cuCtxSetCurrent(currCudaCtx));
      }

      glDeleteBuffers(1, &_bufferId);
    }
  }

  CudaHandle getCudaHandlePreQuery() {
    // Handling the state of the buffer since the GL VBO needs to be mapped/unmapped to/from a CUDA buffer.
    // Managing the state ensures that the mapping/unmapping is done in the appropriate order.
    RUNTIME_EX_ASSERT(!_isActive, "Query result buffer is already in use. Cannot access cuda handle.");

    // lazy load
    _initBuffer();

    // now map the buffer for cuda
    CUgraphicsResource cudaRsrc = _getCudaGraphicsResource(true);
    _mapCudaGraphicsResource(cudaRsrc);

    size_t num_bytes;

    CUdeviceptr devPtr;
    // checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, _cudaResource));
    checkCudaErrors(cuGraphicsResourceGetMappedPointer(&devPtr, &num_bytes, cudaRsrc));

    RUNTIME_EX_ASSERT(num_bytes == _numTotalBytes,
                      "QueryResultVertexBuffer: couldn't successfully map all " + std::to_string(_numTotalBytes) +
                          " bytes. Was only able to map " + std::to_string(num_bytes) + " bytes.");

    _isActive = true;

    return CudaHandle(reinterpret_cast<void*>(devPtr), _numTotalBytes);
  }

  void updatePostQuery(const BufferLayoutShPtr& bufferLayout, const int numRows) {
    // TODO(croot): fill this function in. Should be called after the query is completed
    // and this buffer is filled with data. We just need to know what's in that data.

    RUNTIME_EX_ASSERT(_isActive, "Query result buffer has not been prepped for a query. Cannot set data post query.");

    _isActive = false;

    // unmap buffer object
    CUgraphicsResource cudaRsrc = _getCudaGraphicsResource();
    _unmapCudaGraphicsResource(cudaRsrc);

    _layoutPtr = bufferLayout;
    _size = numRows;
  }

 private:
  static void checkCudaErrors(CUresult result) {
    RUNTIME_EX_ASSERT(result == CUDA_SUCCESS, "CUDA error code=" + std::to_string(static_cast<unsigned int>(result)));
  }

  bool _isActive;
  unsigned int _numTotalBytes;

  // CUgraphicsResource _cudaResource;

  std::unordered_map<CUcontext, CUgraphicsResource> _cudaResourceMap;
  std::unordered_set<CUgraphicsResource> _mappedCudaResources;

  CUgraphicsResource _getCudaGraphicsResource(bool registerResource = false) {
    // CUcontext* currCudaCtx=nullptr;
    CUcontext currCudaCtx;

    checkCudaErrors(cuCtxGetCurrent(&currCudaCtx));

    // TODO(croot): convert these checks to log/throw errors
    CHECK(currCudaCtx != nullptr);
    CHECK(_bufferId);

    const auto itr = _cudaResourceMap.find(currCudaCtx);
    if (itr == _cudaResourceMap.end()) {
      CHECK(registerResource);

      CUgraphicsResource rsrc;
      checkCudaErrors(cuGraphicsGLRegisterBuffer(&rsrc, _bufferId, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
      _cudaResourceMap.insert(std::make_pair(currCudaCtx, rsrc));
      return rsrc;
    }

    return itr->second;
  }

  void _mapCudaGraphicsResource(CUgraphicsResource& rsrc) {
    if (_mappedCudaResources.find(rsrc) == _mappedCudaResources.end()) {
      checkCudaErrors(cuGraphicsMapResources(1, &rsrc, 0));
      _mappedCudaResources.insert(rsrc);
    }  // already mapped otherwise
  }

  void _unmapCudaGraphicsResource(CUgraphicsResource& rsrc) {
    const auto itr = _mappedCudaResources.find(rsrc);
    if (itr != _mappedCudaResources.end()) {
      checkCudaErrors(cuGraphicsUnmapResources(1, &rsrc, 0));
      _mappedCudaResources.erase(itr);
    }
    // TODO(croot): throw an error if nothing to unmap?
  }

  void _initBuffer() {
    if (!_bufferId) {
      BaseVertexBuffer::_initBuffer();

      // don't mess with the current state
      // TODO(croot): Apply some kind of push-pop state system
      GLint currArrayBuf;
      glGetIntegerv(_getBufferBinding(_target), &currArrayBuf);

      glBindBuffer(_target, _bufferId);
      glBufferData(_target, _numTotalBytes, 0, _usage);

      // restore the state
      glBindBuffer(_target, currArrayBuf);
    }
  }
};

typedef std::unique_ptr<QueryResultVertexBuffer> QueryResultVertexBufferUqPtr;
typedef std::shared_ptr<QueryResultVertexBuffer> QueryResultVertexBufferShPtr;

}  // namespace MapD_Renderer

#endif  // QUERY_RESULT_VERTEX_BUFFER_H_
