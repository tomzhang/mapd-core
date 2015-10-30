#ifndef QUERY_RESULT_VERTEX_BUFFER_H_
#define QUERY_RESULT_VERTEX_BUFFER_H_

#include "BufferLayout.h"
#include "Shader.h"
#include "VertexBuffer.h"
#include <GL/glew.h>
#include <glog/logging.h>

// CUDA libs
// CROOT: CUDA COMMENT
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <cstdint>

namespace MapD_Renderer {

struct CudaHandle {
  // CROOT: CUDA COMMENT
  void* handle;

  // GLuint handle;
  unsigned int numBytes;

  // CROOT: CUDA COMMENT
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
        _cudaResource(nullptr) {}

  ~QueryResultVertexBuffer() {
    if (_bufferId) {
      // TODO(croot): check for cuda errors?

      // CROOT: CUDA COMMENT
      // if (_isActive) {
      //   checkCudaErrors(cudaGraphicsUnmapResources(1, &_cudaResource, 0));
      // }

      // TODO(croot)
      // this keeps erroring out and I'm not sure why
      // but comment out until I do
      // checkCudaErrors(cudaGraphicsUnregisterResource(_cudaResource));

      // NOTE: let the base class destroy _bufferId
      // glDeleteBuffers(1, &_bufferId);
    }
  }

  // bool hasAttribute(const std::string& attrName) {
  //   if (!_layoutPtr) {
  //     return false;
  //   }
  //   return _layoutPtr->hasAttribute(attrName);
  // }

  // TypeGLShPtr getAttributeTypeGL(const std::string& attrName) {
  //   if (!_layoutPtr) {
  //     throw std::runtime_error(
  //         "The query buffer has not been properly initialized. Cannot retrieve the Attribute GL Type of " + attrName
  //         +
  //         ".");
  //   }
  //   return _layoutPtr->getAttributeTypeGL(attrName);
  // }

  void setBufferLayout(const BufferLayoutShPtr& layoutPtr) { _layoutPtr = layoutPtr; }

  CudaHandle getCudaHandlePreQuery() {
    // Handling the state of the buffer since the GL VBO needs to be mapped/unmapped to/from a CUDA buffer.
    // Managing the state ensures that the mapping/unmapping is done in the appropriate order.
    if (_isActive) {
      std::runtime_error err("Query result buffer is already in use. Cannot access cuda handle.");
      LOG(ERROR) << err.what();
      throw err;
    }

    // lazy load
    _initBuffer();

    // now map the buffer for cuda
    // TODO(croot) - check for cuda errors

    // CROOT: CUDA COMMENT
    checkCudaErrors(cudaGraphicsMapResources(1, &_cudaResource, 0));

    void* cudaPtr;
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &num_bytes, _cudaResource));

    if (num_bytes != _numTotalBytes) {
      std::runtime_error err("QueryResultVertexBuffer: couldn't successfully map all " +
                             std::to_string(_numTotalBytes) + " bytes. Was only able to map " +
                             std::to_string(num_bytes) + " bytes.");
      LOG(ERROR) << err.what();
      throw err;
    }

    _isActive = true;

    // return CudaHandle(_bufferId, _numTotalBytes);
    return CudaHandle(cudaPtr, _numTotalBytes);
  }

  void updatePostQuery(const BufferLayoutShPtr& bufferLayout, const int numRows) {
    // TODO(croot): fill this function in. Should be called after the query is completed
    // and this buffer is filled with data. We just need to know what's in that data.

    if (!_isActive) {
      std::runtime_error err("Query result buffer has not been prepped for a query. Cannot set data post query.");
      LOG(ERROR) << err.what();
      throw err;
    }

    _isActive = false;

    // unmap buffer object
    // TODO(croot) check for cuda errors

    // CROOT: CUDA COMMENT
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cudaResource, 0));

    _layoutPtr = bufferLayout;
    _size = numRows;
  }

 private:
  // CROOT: CUDA COMMENT
  static void checkCudaErrors(cudaError_t result) {
    if (result) {
      fprintf(stderr, "CUDA error code=%d\n", static_cast<unsigned int>(result));
      assert(false);
    }
  }

  bool _isActive;
  unsigned int _numTotalBytes;

  // CROOT: CUDA COMMENT
  struct cudaGraphicsResource* _cudaResource;

  void _initBuffer() {
    if (!_bufferId) {
      // glGenBuffers(1, &_bufferId);
      BaseVertexBuffer::_initBuffer();

      // don't mess with the current state
      // TODO(croot): Apply some kind of push-pop state system
      GLint currArrayBuf;
      glGetIntegerv(_getBufferBinding(_target), &currArrayBuf);

      glBindBuffer(_target, _bufferId);
      glBufferData(_target, _numTotalBytes, 0, _usage);

      // CROOT: CUDA COMMENT
      checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cudaResource, _bufferId, cudaGraphicsRegisterFlagsWriteDiscard));

      // restore the state
      glBindBuffer(_target, currArrayBuf);
    }
  }
};

typedef std::unique_ptr<QueryResultVertexBuffer> QueryResultVertexBufferUqPtr;
typedef std::shared_ptr<QueryResultVertexBuffer> QueryResultVertexBufferShPtr;

}  // namespace MapD_Renderer

#endif  // QUERY_RESULT_VERTEX_BUFFER_H_
