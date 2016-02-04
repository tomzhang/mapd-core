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

#ifdef HAVE_CUDA
struct CudaHandle {
  void* handle;

  // GLuint handle;
  unsigned int numBytes;

  CudaHandle(void* handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}

  // CudaHandle(GLuint handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}
};
#endif

// TODO(croot): create a base VBO class that both the QueryResultVertexBuffer and VertexBuffer classes can
// inherit from
class QueryResultVertexBuffer {
 public:
  explicit QueryResultVertexBuffer(Rendering::GL::GLRenderer* renderer,
                                   size_t numBytes,
                                   GLenum usage = GL_DYNAMIC_COPY);

  ~QueryResultVertexBuffer();

#ifdef HAVE_CUDA
  CudaHandle getCudaHandlePreQuery();
#endif

  void updatePostQuery(const Rendering::GL::Resources::GLBufferLayoutShPtr& bufferLayout, size_t numRows);

 private:
  Rendering::GL::Resources::GLVertexBufferShPtr _vbo;
  bool _isActive;

#ifdef HAVE_CUDA
  // CUgraphicsResource _cudaResource;
  std::unordered_map<CUcontext, CUgraphicsResource> _cudaResourceMap;
  std::unordered_set<CUgraphicsResource> _mappedCudaResources;

  static void checkCudaErrors(CUresult result);

  CUgraphicsResource _getCudaGraphicsResource(bool registerResource = false);

  void _mapCudaGraphicsResource(CUgraphicsResource& rsrc);

  void _unmapCudaGraphicsResource(CUgraphicsResource& rsrc);
#endif  // HAVE_CUDA

  void _initBuffer(Rendering::GL::GLRenderer* renderer, size_t numBytes, GLenum usage);
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRESULTVERTEXBUFFER_H_
