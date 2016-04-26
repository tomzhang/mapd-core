#ifndef QUERYRENDERER_INTEROP_QUERYBUFFER_H_
#define QUERYRENDERER_INTEROP_QUERYBUFFER_H_

#include "../Types.h"
#include "../QueryDataLayout.h"
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Enums.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <Rendering/Renderer/GL/Resources/GLBufferLayout.h>
#include <Rendering/Renderer/GL/Resources/GLVertexBuffer.h>
#include <Rendering/Renderer/GL/Resources/GLIndexBuffer.h>
#include <Rendering/Renderer/GL/Resources/GLUniformBuffer.h>
#include <Rendering/Renderer/GL/Resources/GLIndirectDrawBuffer.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cudaGL.h>
#endif  // HAVE_CUDA

#include <unordered_map>
#include <unordered_set>

namespace QueryRenderer {

struct CudaHandle {
  void* handle;
  unsigned int numBytes;

  CudaHandle() : handle(nullptr), numBytes(0) {}
  CudaHandle(void* handle, unsigned int numBytes) : handle(handle), numBytes(numBytes) {}
};

struct PolyTableByteData {
  size_t numVertBytes;
  size_t numIndexBytes;
  size_t numLineLoopBytes;
  size_t numPolyBytes;
  size_t numDataBytes;

  // PolyTableByteData() : numVertBytes(0), numIndexBytes(0), numLineLoopBytes(0), numPolyBytes(0), numDataBytes(0) {}
};

struct PolyTableDataInfo {
  size_t numVerts;
  size_t numTris;
  size_t numLineLoops;
  size_t numPolys;
  size_t numDataBytes;

  PolyTableDataInfo() : numVerts(0), numTris(0), numLineLoops(0), numPolys(0), numDataBytes(0) {}
};

struct PolyCudaHandles {
  CudaHandle verts;
  CudaHandle polyIndices;
  CudaHandle lineDrawStruct;
  CudaHandle polyDrawStruct;
  CudaHandle perRowData;

  PolyCudaHandles() {}
};

class QueryBuffer {
 public:
  enum class BufType { QUERY_BUFFER = 0, QUERY_RESULT_BUFFER };

  explicit QueryBuffer(BufType type = BufType::QUERY_BUFFER);

  virtual ~QueryBuffer();

  size_t getNumUsedBytes() const { return _usedBytes; }
  size_t numBytes() const { return (_bufRsrc ? _bufRsrc->numBytes() : 0); }
  void reset() { _usedBytes = 0; }

  BufType getType() const { return _type; }
  ::Rendering::GL::Resources::GLResourceType getGLResourceType() const {
    CHECK(_bufRsrc);
    return _bufRsrc->getResourceType();
  }

  ::Rendering::GL::Resources::GLBaseBufferShPtr getGLBufferPtr() const { return _bufRsrc; }

  virtual ::Rendering::GL::Resources::GLBufferAttrType getAttributeType(const std::string& attrName) const {
    THROW_RUNTIME_EX("QueryBuffer::getAttributeType() is undefined for " + std::to_string(static_cast<int>(getType())));
  }

  virtual ::Rendering::GL::TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const {
    THROW_RUNTIME_EX("QueryBuffer::getAttributeTypeGL() is undefined for " +
                     std::to_string(static_cast<int>(getType())));
  }

#ifdef HAVE_CUDA
  CudaHandle getCudaHandlePreQuery(bool useAllAllocatedBytes = false);
#endif

  void updatePostQuery(size_t numUsedBytes);

  void resize(size_t numBytes);

 protected:
  Rendering::GL::Resources::GLBaseBufferShPtr _bufRsrc;

  size_t _getGpuId() const;
  void _setUsedBytes(size_t usedBytes) { _usedBytes = usedBytes; }

#ifdef HAVE_CUDA
  void _initCudaGraphicsResource();
  void _removeCudaGraphicsResource();
#endif

 private:
  BufType _type;
  bool _isActive;
  size_t _usedBytes;

#ifdef HAVE_CUDA
  std::unordered_map<CUcontext, CUgraphicsResource> _cudaResourceMap;
  std::unordered_set<CUgraphicsResource> _mappedCudaResources;

  void checkCudaErrors(CUresult result, const char* filename, int lineno);

  CUgraphicsResource _getCudaGraphicsResource(bool registerResource = false);

  void _mapCudaGraphicsResource(CUgraphicsResource& rsrc);
  void _unmapCudaGraphicsResource(CUgraphicsResource& rsrc);
#endif  // HAVE_CUDA
};

template <typename GLBufferType, typename GLLayoutType = ::Rendering::GL::Resources::GLBaseBufferLayout>
class QueryLayoutBuffer : public QueryBuffer {
 public:
  explicit QueryLayoutBuffer(QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER) : QueryBuffer(type) {}

  virtual ~QueryLayoutBuffer() {}

  bool hasAttribute(const std::string& attrName) const {
    CHECK(_bufRsrc);
    return _getGLResource()->hasAttribute(attrName);
  }

  size_t numItems() const {
    CHECK(_bufRsrc);
    return _getGLResource()->numItems();
  }

  ::Rendering::GL::Resources::GLBufferAttrType getAttributeType(const std::string& attrName) const final {
    CHECK(_bufRsrc);
    return _getGLResource()->getAttributeType(attrName);
  }

  ::Rendering::GL::TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const final {
    CHECK(_bufRsrc);
    return _getGLResource()->getAttributeTypeGL(attrName);
  }

  void setBufferLayout(const std::shared_ptr<GLLayoutType>& bufferLayout) {
    size_t bytesPerVertex = bufferLayout->getNumBytesPerItem();
    size_t usedBytes = getNumUsedBytes();
    RUNTIME_EX_ASSERT(usedBytes % bytesPerVertex == 0,
                      "QueryLayoutBuffer " + std::to_string(_getGpuId()) + ": Buffer layout bytes-per-vertex " +
                          std::to_string(bytesPerVertex) +
                          " does not align with the number of used bytes in the buffer: " + std::to_string(usedBytes) +
                          ".");

    CHECK(_bufRsrc);
    _getGLResource()->setBufferLayout(bufferLayout, usedBytes / bytesPerVertex);
  }

  void bufferData(void* data, size_t numItems, size_t numBytesPerItem) {
    CHECK(_bufRsrc);
    _getGLResource()->bufferData(data, numItems, numBytesPerItem);
    _setUsedBytes(numItems * numBytesPerItem);
  }

  QueryDataLayoutShPtr getQueryDataLayout() const { return _queryDataLayoutPtr; }

  void setQueryDataLayout(const QueryDataLayoutShPtr& queryDataLayoutPtr) {
    CHECK(queryDataLayoutPtr && getType() == QueryBuffer::BufType::QUERY_RESULT_BUFFER);
    _queryDataLayoutPtr = queryDataLayoutPtr;
    setBufferLayout(queryDataLayoutPtr->convertToBufferLayout());
  }

 protected:
  std::shared_ptr<GLBufferType> _getGLResource() const { return std::dynamic_pointer_cast<GLBufferType>(_bufRsrc); }

 private:
  virtual void _initLayoutBuffer(Rendering::GL::GLRenderer* renderer,
                                 Rendering::GL::Resources::BufferAccessType accessType,
                                 Rendering::GL::Resources::BufferAccessFreq accessFreq,
                                 const std::shared_ptr<GLLayoutType>& layoutPtr) = 0;

  QueryDataLayoutShPtr _queryDataLayoutPtr;
};

class QueryVertexBuffer : public QueryLayoutBuffer<::Rendering::GL::Resources::GLVertexBuffer> {
 public:
  explicit QueryVertexBuffer(QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryVertexBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      Rendering::GL::Resources::BufferAccessType accessType =
          Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      Rendering::GL::Resources::BufferAccessFreq accessFreq = Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryVertexBuffer(
      Rendering::GL::GLRenderer* renderer,
      const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr,
      Rendering::GL::Resources::BufferAccessType accessType =
          Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      Rendering::GL::Resources::BufferAccessFreq accessFreq = Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  ~QueryVertexBuffer();

  ::Rendering::GL::Resources::GLVertexBufferShPtr getGLVertexBufferPtr() const { return _getGLResource(); }

  int numVertices() const {
    auto rsrc = getGLVertexBufferPtr();
    return (rsrc ? rsrc->numVertices() : 0);
  }

 private:
  void _initBuffer(Rendering::GL::GLRenderer* renderer,
                   Rendering::GL::Resources::BufferAccessType accessType,
                   Rendering::GL::Resources::BufferAccessFreq accessFreq,
                   size_t numBytes);

  void _initLayoutBuffer(Rendering::GL::GLRenderer* renderer,
                         Rendering::GL::Resources::BufferAccessType accessType,
                         Rendering::GL::Resources::BufferAccessFreq accessFreq,
                         const Rendering::GL::Resources::GLBufferLayoutShPtr& layoutPtr) final;
};

class QueryResultVertexBuffer : public QueryVertexBuffer {
 public:
  explicit QueryResultVertexBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      Rendering::GL::Resources::BufferAccessType accessType = Rendering::GL::Resources::BufferAccessType::COPY,
      Rendering::GL::Resources::BufferAccessFreq accessFreq = Rendering::GL::Resources::BufferAccessFreq::DYNAMIC);

  ~QueryResultVertexBuffer();
};

class QueryIndexBuffer : public QueryBuffer {
 public:
  explicit QueryIndexBuffer(QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndexBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      ::Rendering::GL::Resources::GLIndexBuffer::IndexType indexType =
          ::Rendering::GL::Resources::GLIndexBuffer::IndexType::UNSIGNED_INT,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndexBuffer(
      ::Rendering::GL::GLRenderer* renderer,
      const std::vector<unsigned char>& items,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndexBuffer(
      ::Rendering::GL::GLRenderer* renderer,
      const std::vector<unsigned short>& items,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndexBuffer(
      ::Rendering::GL::GLRenderer* renderer,
      const std::vector<unsigned int>& items,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  ~QueryIndexBuffer();

  ::Rendering::GL::Resources::GLIndexBufferShPtr getGLIndexBufferPtr() const {
    return std::dynamic_pointer_cast<::Rendering::GL::Resources::GLIndexBuffer>(_bufRsrc);
  }

  size_t numItems() const {
    auto rsrc = getGLIndexBufferPtr();
    return (rsrc ? rsrc->numItems() : 0);
  }

 private:
  void _initBuffer(::Rendering::GL::GLRenderer* renderer,
                   ::Rendering::GL::Resources::GLIndexBuffer::IndexType indexType,
                   ::Rendering::GL::Resources::BufferAccessType accessType,
                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                   size_t numBytes);
};

class QueryResultIndexBuffer : public QueryIndexBuffer {
 public:
  explicit QueryResultIndexBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      ::Rendering::GL::Resources::GLIndexBuffer::IndexType indexType =
          ::Rendering::GL::Resources::GLIndexBuffer::IndexType::UNSIGNED_INT,
      ::Rendering::GL::Resources::BufferAccessType accessType = ::Rendering::GL::Resources::BufferAccessType::COPY,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::DYNAMIC);

  ~QueryResultIndexBuffer();
};

class QueryUniformBuffer : public QueryLayoutBuffer<::Rendering::GL::Resources::GLUniformBuffer,
                                                    ::Rendering::GL::Resources::GLShaderBlockLayout> {
 public:
  explicit QueryUniformBuffer(QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryUniformBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryUniformBuffer(
      ::Rendering::GL::GLRenderer* renderer,
      const ::Rendering::GL::Resources::GLShaderBlockLayoutShPtr& layoutPtr,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  ~QueryUniformBuffer();

  ::Rendering::GL::Resources::GLUniformBufferShPtr getGLUniformBufferPtr() const {
    return std::dynamic_pointer_cast<::Rendering::GL::Resources::GLUniformBuffer>(_bufRsrc);
  }

 private:
  void _initBuffer(::Rendering::GL::GLRenderer* renderer,
                   ::Rendering::GL::Resources::BufferAccessType accessType,
                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                   size_t numBytes);

  void _initLayoutBuffer(Rendering::GL::GLRenderer* renderer,
                         Rendering::GL::Resources::BufferAccessType accessType,
                         Rendering::GL::Resources::BufferAccessFreq accessFreq,
                         const ::Rendering::GL::Resources::GLShaderBlockLayoutShPtr& layoutPtr) final;
};

class QueryResultUniformBuffer : public QueryUniformBuffer {
 public:
  explicit QueryResultUniformBuffer(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      Rendering::GL::Resources::BufferAccessType accessType = Rendering::GL::Resources::BufferAccessType::COPY,
      Rendering::GL::Resources::BufferAccessFreq accessFreq = Rendering::GL::Resources::BufferAccessFreq::DYNAMIC);

  ~QueryResultUniformBuffer();
};

class QueryIndirectVbo : public QueryBuffer {
 public:
  explicit QueryIndirectVbo(QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndirectVbo(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndirectVbo(
      ::Rendering::GL::GLRenderer* renderer,
      const std::vector<::Rendering::GL::Resources::IndirectDrawVertexData>& items,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  ~QueryIndirectVbo();

  ::Rendering::GL::Resources::GLIndirectDrawVertexBufferShPtr getGLIndirectVboPtr() const {
    return std::dynamic_pointer_cast<::Rendering::GL::Resources::GLIndirectDrawVertexBuffer>(_bufRsrc);
  }

  size_t numItems() const {
    auto rsrc = getGLIndirectVboPtr();
    return (rsrc ? rsrc->numItems() : 0);
  }

 private:
  void _initBuffer(::Rendering::GL::GLRenderer* renderer,
                   ::Rendering::GL::Resources::BufferAccessType accessType,
                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                   size_t numBytes);
};

class QueryResultIndirectVbo : public QueryIndirectVbo {
 public:
  explicit QueryResultIndirectVbo(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      ::Rendering::GL::Resources::BufferAccessType accessType = ::Rendering::GL::Resources::BufferAccessType::COPY,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::DYNAMIC);

  ~QueryResultIndirectVbo();
};

class QueryIndirectIbo : public QueryBuffer {
 public:
  explicit QueryIndirectIbo(QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndirectIbo(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  explicit QueryIndirectIbo(
      ::Rendering::GL::GLRenderer* renderer,
      const std::vector<::Rendering::GL::Resources::IndirectDrawIndexData>& items,
      ::Rendering::GL::Resources::BufferAccessType accessType =
          ::Rendering::GL::Resources::BufferAccessType::READ_AND_WRITE,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::STATIC,
      QueryBuffer::BufType type = QueryBuffer::BufType::QUERY_BUFFER);

  ~QueryIndirectIbo();

  ::Rendering::GL::Resources::GLIndirectDrawIndexBufferShPtr getGLIndirectIboPtr() const {
    return std::dynamic_pointer_cast<::Rendering::GL::Resources::GLIndirectDrawIndexBuffer>(_bufRsrc);
  }

  size_t numItems() const {
    auto rsrc = getGLIndirectIboPtr();
    return (rsrc ? rsrc->numItems() : 0);
  }

 private:
  void _initBuffer(::Rendering::GL::GLRenderer* renderer,
                   ::Rendering::GL::Resources::BufferAccessType accessType,
                   ::Rendering::GL::Resources::BufferAccessFreq accessFreq,
                   size_t numBytes);
};

class QueryResultIndirectIbo : public QueryIndirectIbo {
 public:
  explicit QueryResultIndirectIbo(
      Rendering::GL::GLRenderer* renderer,
      size_t numBytes,
      ::Rendering::GL::Resources::BufferAccessType accessType = ::Rendering::GL::Resources::BufferAccessType::COPY,
      ::Rendering::GL::Resources::BufferAccessFreq accessFreq = ::Rendering::GL::Resources::BufferAccessFreq::DYNAMIC);

  ~QueryResultIndirectIbo();
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_INTEROP_QUERYBUFFER_H_
