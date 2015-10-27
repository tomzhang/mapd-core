#ifndef QUERY_RESULT_VERTEX_BUFFER_H_
#define QUERY_RESULT_VERTEX_BUFFER_H_

#include "BufferLayout.h"
#include "Shader.h"
#include <GL/glew.h>

// CUDA libs
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace MapD_Renderer {

struct CudaHandle {
  unsigned int numBytes;
};

// TODO(croot): create a base VBO class that both the QueryResultVertexBuffer and VertexBuffer classes can
// inherit from
class QueryResultVertexBuffer {
 public:
  typedef std::shared_ptr<BaseBufferLayout> BufferLayoutShPtr;

  explicit QueryResultVertexBuffer(unsigned int numBytes,
                                   GLenum target = GL_ARRAY_BUFFER,
                                   GLenum usage = GL_DYNAMIC_COPY)
      : _numTotalBytes(numBytes),
        _numRows(0),
        _bufferId(0),
        _cudaResource(nullptr),
        _target(target),
        _usage(usage),
        _layoutPtr(nullptr) {}

  ~QueryResultVertexBuffer() {
    if (_bufferId) {
      // TODO(croot): check for cuda errors?
      cudaGraphicsUnregisterResource(_cudaResource);
      glDeleteBuffers(1, &_bufferId);
    }
  }

  bool hasAttribute(const std::string& attrName) {
    if (!_layoutPtr) {
      return false;
    }
    return _layoutPtr->hasAttribute(attrName);
  }

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) {
    if (!_layoutPtr) {
      throw std::runtime_error(
          "The query buffer has not been properly initialized. Cannot retrieve the Attribute GL Type of " + attrName +
          ".");
    }
    return _layoutPtr->getAttributeTypeGL(attrName);
  }

  void setBufferLayout(const std::shared_ptr<BaseBufferLayout>& layoutPtr) { _layoutPtr = layoutPtr; }

  CudaHandle prepForQuery() {
    // lazy load
    _initBuffer();

    // now map the buffer for cuda
    // TODO(croot) - check for cuda errors
    cudaGraphicsMapResources(1, &_cudaResource, 0);

    void* cudaPtr;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer(&cudaPtr, &num_bytes, _cudaResource);

    return CudaHandle();
  }

  void initializeLayoutPostQuery() {
    // TODO(croot): fill this function in. Should be called after the query is completed
    // and this buffer is filled with data. We just need to know what's in that data.

    // unmap buffer object
    // TODO(croot) check for cuda errors
    cudaGraphicsUnmapResources(1, &_cudaResource, 0);

    // assuming interleaved right now
    _layoutPtr.reset(new InterleavedBufferLayout());
  }

  int size() const { return _numRows; }

  void bindToRenderer(Shader* activeShader, const std::string& attr = "", const std::string& shaderAttr = "") {
    if (!_bufferId) {
      throw std::runtime_error("Cannot bind vertex buffer. It has not been initialized with data.");
    } else if (!_layoutPtr) {
      throw std::runtime_error("Cannot bind vertex buffer. It does not have a defined layout.");
    }
    glBindBuffer(_target, _bufferId);
    _layoutPtr->bindToRenderer(activeShader, size(), attr, shaderAttr);
  }

 private:
  static GLenum _getBufferBinding(GLenum target) {
    typedef std::unordered_map<GLenum, GLenum> BufferBindingMap;
    static const BufferBindingMap bufferBindings = {{GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING},
                                                    {GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING}};

    BufferBindingMap::const_iterator itr;

    if ((itr = bufferBindings.find(target)) == bufferBindings.end()) {
      throw std::runtime_error(std::to_string(target) + " is not a valid opengl buffer target");
    }

    return itr->first;
  }

  unsigned int _numTotalBytes;
  unsigned int _numRows;
  GLuint _bufferId;
  struct cudaGraphicsResource* _cudaResource;
  GLenum _target;
  GLenum _usage;
  BufferLayoutShPtr _layoutPtr;

  void _initBuffer() {
    if (!_bufferId) {
      glGenBuffers(1, &_bufferId);

      // don't mess with the current state
      // TODO(croot): Apply some kind of push-pop state system
      // GLint currArrayBuf;
      // glGetIntegerv(_getBufferBinding(_target), &currArrayBuf);

      glBindBuffer(_target, _bufferId);
      glBufferData(_target, _numTotalBytes, 0, _usage);

      // restore the state
      // glBindBuffer(_target, currArrayBuf);
    }
  }
};

typedef std::unique_ptr<QueryResultVertexBuffer> QueryResultVertexBufferUqPtr;
typedef std::shared_ptr<QueryResultVertexBuffer> QueryResultVertexBufferShPtr;

}  // namespace MapD_Renderer

#endif  // QUERY_RESULT_VERTEX_BUFFER_H_
