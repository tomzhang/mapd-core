#ifndef QUERYRENDERER_QUERYRESULTVERTEXBUFFER_H_
#define QUERYRENDERER_QUERYRESULTVERTEXBUFFER_H_

#include <Rendering/Renderer/GL/Resources/GLVertexBuffer.h>
#include <Rendering/Renderer/GL/Types.h>

#include <GL/glew.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cudaGL.h>
#endif  // HAVE_CUDA

#include <unordered_set>

namespace QueryRenderer {

class QueryRenderManager;

struct CudaHandle {
  void* handle;
  unsigned int numBytes;

  CudaHandle(void* handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}
};

class QueryVertexBuffer {
 public:
  enum class VboType { QUERY_VBO = 0, QUERY_RESULT_VBO };

  explicit QueryVertexBuffer(VboType type = VboType::QUERY_VBO);

  explicit QueryVertexBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      Rendering::GL::Resources::BufferAccessType accessType =
          Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      Rendering::GL::Resources::BufferAccessFreq accessFreq = Rendering::GL::Resources::BufferAccessFreq::STATIC,
      VboType type = VboType::QUERY_VBO);
  explicit QueryVertexBuffer(
      Rendering::GL::GLRenderer* renderer,
      const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr,
      Rendering::GL::Resources::BufferAccessType accessType =
          Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      Rendering::GL::Resources::BufferAccessFreq accessFreq = Rendering::GL::Resources::BufferAccessFreq::STATIC,
      VboType type = VboType::QUERY_VBO);

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
                   Rendering::GL::Resources::BufferAccessType accessType,
                   Rendering::GL::Resources::BufferAccessFreq accessFreq,
                   size_t numBytes = 0,
                   const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr = nullptr);
};

// TODO(croot): create a base VBO class that both the QueryResultVertexBuffer and VertexBuffer classes can
// inherit from
class QueryResultVertexBuffer : public QueryVertexBuffer {
 public:
  explicit QueryResultVertexBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      Rendering::GL::Resources::BufferAccessType accessType = Rendering::GL::Resources::BufferAccessType::COPY,
      Rendering::GL::Resources::BufferAccessFreq accessFreq = Rendering::GL::Resources::BufferAccessFreq::DYNAMIC);

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

  // temporarily making the checkCudaErrors
  // static void checkCudaErrors(CUresult result);
  void checkCudaErrors(CUresult result, const char* filename, int lineno);

  void _initCudaGraphicsResource();
  CUgraphicsResource _getCudaGraphicsResource();

  void _mapCudaGraphicsResource(CUgraphicsResource& rsrc);

  void _unmapCudaGraphicsResource(CUgraphicsResource& rsrc);
#endif  // HAVE_CUDA

  friend class ::QueryRenderer::QueryRenderManager;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYRESULTVERTEXBUFFER_H_
