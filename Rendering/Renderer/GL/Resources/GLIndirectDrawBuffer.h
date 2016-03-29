#ifndef RENDERING_GL_RESOURCES_GLINDIRECTDRAWBUFFER_H_
#define RENDERING_GL_RESOURCES_GLINDIRECTDRAWBUFFER_H_

#include "Types.h"
#include "GLBaseBuffer.h"
#include "../TypeGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLIndirectDrawVertexBuffer;
class GLIndirectDrawIndexBuffer;

template <typename T>
class GLBaseIndirectDrawBuffer : public GLBaseBuffer {
 public:
  virtual ~GLBaseIndirectDrawBuffer();

  GLResourceType getResourceType() const final { return _rsrcType; }
  size_t numItems() const { return _numItems; }
  size_t getStructByteSize() const { return sizeof(T); }

  // virtual GLBufferType getBufferType() const;

  void bufferData(const std::vector<T>& indirectDrawData) {
    GLBaseBuffer::bufferData(&indirectDrawData[0], indirectDrawData.size() * sizeof(T));
    _numItems = indirectDrawData.size();
  }

  // void setDataAt(size_t idx, const T& indirectData);

 private:
  explicit GLBaseIndirectDrawBuffer(const RendererWkPtr& rendererPtr,
                                    GLResourceType rsrcType,
                                    GLBufferType bufferType,
                                    BufferAccessType accessType,
                                    BufferAccessFreq accessFreq)
      : GLBaseBuffer(rendererPtr, bufferType, GL_DRAW_INDIRECT_BUFFER, accessType, accessFreq), _rsrcType(rsrcType) {}

  explicit GLBaseIndirectDrawBuffer(const RendererWkPtr& rendererPtr,
                                    GLResourceType rsrcType,
                                    GLBufferType bufferType,
                                    size_t numBytes,
                                    BufferAccessType accessType,
                                    BufferAccessFreq accessFreq)
      : GLBaseIndirectDrawBuffer(rendererPtr, rsrcType, bufferType, accessType, accessFreq) {
    RUNTIME_EX_ASSERT(numBytes % sizeof(T) == 0,
                      "Cannot allocate an indirect draw buffer of type " + to_string(rsrcType) + " with " +
                          std::to_string(numBytes) + " bytes. The size of the buffer must be a multiple of " +
                          std::to_string(sizeof(T)));

    GLBaseBuffer::bufferData(nullptr, numBytes);
  }

  explicit GLBaseIndirectDrawBuffer(const RendererWkPtr& rendererPtr,
                                    GLResourceType rsrcType,
                                    GLBufferType bufferType,
                                    const std::vector<T>& items,
                                    BufferAccessType accessType,
                                    BufferAccessFreq accessFreq)
      : GLBaseIndirectDrawBuffer(rendererPtr, rsrcType, bufferType, accessType, accessFreq) {
    bufferData(items);
  }

  void _makeEmpty() final { _numItems = 0; }

  size_t _numItems;
  GLResourceType _rsrcType;

  friend class ::Rendering::GL::Resources::GLIndirectDrawVertexBuffer;
  friend class ::Rendering::GL::Resources::GLIndirectDrawIndexBuffer;
};

typedef struct {
  size_t count;
  size_t instanceCount;
  size_t first;
  size_t baseInstance;
} IndirectDrawVertexData;

typedef struct {
  size_t count;
  size_t instanceCount;
  size_t firstIndex;
  size_t baseVertex;
  size_t baseInstance;
} IndirectDrawIndexData;

class GLIndirectDrawVertexBuffer : public GLBaseIndirectDrawBuffer<IndirectDrawVertexData> {
 public:
  ~GLIndirectDrawVertexBuffer() {}

 private:
  explicit GLIndirectDrawVertexBuffer(const RendererWkPtr& rendererPtr,
                                      BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                                      BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndirectDrawVertexBuffer(const RendererWkPtr& rendererPtr,
                                      size_t numBytes,
                                      BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                                      BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndirectDrawVertexBuffer(const RendererWkPtr& rendererPtr,
                                      const std::vector<IndirectDrawVertexData>& items,
                                      BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                                      BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  friend class ::Rendering::GL::GLResourceManager;
};

class GLIndirectDrawIndexBuffer : public GLBaseIndirectDrawBuffer<IndirectDrawIndexData> {
 public:
  ~GLIndirectDrawIndexBuffer() {}

 private:
  explicit GLIndirectDrawIndexBuffer(const RendererWkPtr& rendererPtr,
                                     BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                                     BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndirectDrawIndexBuffer(const RendererWkPtr& rendererPtr,
                                     size_t numBytes,
                                     BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                                     BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndirectDrawIndexBuffer(const RendererWkPtr& rendererPtr,
                                     const std::vector<IndirectDrawIndexData>& items,
                                     BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                                     BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  friend class ::Rendering::GL::GLResourceManager;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLINDIRECTDRAWBUFFER_H_
