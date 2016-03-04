#ifndef QUERYRENDERER_QUERYRESULTVERTEXBUFFER_H_
#define QUERYRENDERER_QUERYRESULTVERTEXBUFFER_H_

#include <Rendering/Renderer/GL/Resources/GLVertexBuffer.h>
#include <Rendering/Renderer/GL/Types.h>

// #include "MapDGL.h"
// #include "QueryRendererError.h"
// #include "BufferLayout.h"
// #include "Shader.h"
// #include "VertexBuffer.h"
#include <GL/glew.h>
// #include <glog/logging.h>

#ifdef HAVE_CUDA
// // CUDA libs
// // #include <cuda_runtime.h>
// // #include <cuda_gl_interop.h>
#include <cuda.h>
#include <cudaGL.h>
#endif  // HAVE_CUDA

// #include <vector>
// #include <memory>
// #include <iostream>
// #include <unordered_map>
// #include <string>
// #include <stdexcept>
// #include <cstdint>
// #include <unordered_map>
#include <unordered_set>

namespace QueryRenderer {

class QueryRenderManager;

#ifdef HAVE_CUDA
struct CudaHandle {
  void* handle;

  // GLuint handle;
  unsigned int numBytes;

  CudaHandle(void* handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}

  // CudaHandle(GLuint handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}
};
#endif

class QueryVertexBuffer {
 public:
  enum class VboType { QUERY_VBO = 0, QUERY_RESULT_VBO };

  explicit QueryVertexBuffer(VboType type = VboType::QUERY_VBO);

  // explicit QueryVertexBuffer(Rendering::GL::GLRenderer* renderer, GLenum usage = GL_STATIC_DRAW);
  explicit QueryVertexBuffer(Rendering::GL::GLRenderer* renderer,
                             size_t numBytes,
                             GLenum usage = GL_STATIC_DRAW,
                             VboType type = VboType::QUERY_VBO);
  explicit QueryVertexBuffer(Rendering::GL::GLRenderer* renderer,
                             const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr,
                             GLenum usage = GL_STATIC_DRAW,
                             VboType type = VboType::QUERY_VBO);

  // template <typename T>
  // explicit QueryVertexBuffer(Rendering::GL::GLRenderer* renderer,
  //                            const std::vector<T>& data,
  //                            const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr,
  //                            GLenum usage = GL_STATIC_DRAW)
  //     : QueryVertexBuffer(renderer, layoutPtr, usage) {
  //   // TODO(croot): validate that the data and the layout align
  //   // NOTE: _size will be set in the bufferData() call
  //   _vbo->bufferData((void*)&data[0], data.size(), sizeof(T));
  // }

  ~QueryVertexBuffer();

  bool hasAttribute(const std::string& attrName) const;
  size_t numItems() const;
  VboType getType() const { return _type; }
  ::Rendering::GL::Resources::GLBufferAttrType getAttributeType(const std::string& attrName) const;
  ::Rendering::GL::TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const;
  ::Rendering::GL::Resources::GLVertexBufferShPtr getGLVertexBufferPtr() const { return _vbo; }

  void bufferData(void* data, size_t numItems, size_t numBytesPerItem);

 protected:
  Rendering::GL::Resources::GLVertexBufferShPtr _vbo;

 private:
  VboType _type;
  void _initBuffer(Rendering::GL::GLRenderer* renderer,
                   GLenum usage,
                   size_t numBytes = 0,
                   const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr = nullptr);
};

// TODO(croot): create a base VBO class that both the QueryResultVertexBuffer and VertexBuffer classes can
// inherit from
class QueryResultVertexBuffer : public QueryVertexBuffer {
 public:
  explicit QueryResultVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                   size_t numBytes,
                                   GLenum usage = GL_DYNAMIC_COPY);

  ~QueryResultVertexBuffer();

  size_t getNumUsedBytes() { return _usedBytes; }

#ifdef HAVE_CUDA
  CudaHandle getCudaHandlePreQuery();
#endif

  void updatePostQuery(size_t numUsedBytes);
  void setBufferLayout(const Rendering::GL::Resources::GLBufferLayoutShPtr& bufferLayout);

 private:
  bool _isActive;
  size_t _usedBytes;
  int _gpuId;

  void reset() { _usedBytes = 0; }

#ifdef HAVE_CUDA
  // CUgraphicsResource _cudaResource;
  std::unordered_map<CUcontext, CUgraphicsResource> _cudaResourceMap;
  std::unordered_set<CUgraphicsResource> _mappedCudaResources;

  static void checkCudaErrors(CUresult result);

  CUgraphicsResource _getCudaGraphicsResource(bool registerResource = false);

  void _mapCudaGraphicsResource(CUgraphicsResource& rsrc);

  void _unmapCudaGraphicsResource(CUgraphicsResource& rsrc);
#endif  // HAVE_CUDA

  friend class ::QueryRenderer::QueryRenderManager;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRESULTVERTEXBUFFER_H_
