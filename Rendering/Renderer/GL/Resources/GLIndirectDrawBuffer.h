#ifndef RENDERING_GL_RESOURCES_GLINDIRECTDRAWBUFFER_H_
#define RENDERING_GL_RESOURCES_GLINDIRECTDRAWBUFFER_H_

#include "Types.h"
#include "GLBaseBuffer.h"
#include "../TypeGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLBaseIndirectDrawBuffer : public GLBaseBuffer {
 public:
  virtual ~GLBaseIndirectDrawBuffer() {}

  size_t numItems() const { return numBytes() / getStructByteSize(); }
  virtual size_t getStructByteSize() const = 0;

 protected:
  explicit GLBaseIndirectDrawBuffer(const RendererWkPtr& rendererPtr,
                                    GLResourceType rsrcType,
                                    GLBufferType bufferType,
                                    BufferAccessType accessType,
                                    BufferAccessFreq accessFreq)
      : GLBaseBuffer(rendererPtr, rsrcType, bufferType, GL_DRAW_INDIRECT_BUFFER, accessType, accessFreq) {}

  void _makeEmpty() final {}
};

template <typename T>
class GLBaseIndirectDrawTemplateBuffer : public GLBaseIndirectDrawBuffer {
 public:
  virtual ~GLBaseIndirectDrawTemplateBuffer() {}

  size_t getStructByteSize() const final { return sizeof(T); }

  void bufferData(const std::vector<T>& indirectDrawData) {
    GLBaseBuffer::bufferData(&indirectDrawData[0], indirectDrawData.size() * sizeof(T));
  }

  void bufferData(const void* data, size_t numBytes, GLenum altTarget = 0) {
    RUNTIME_EX_ASSERT(numBytes % sizeof(T) == 0,
                      "Cannot allocate an indirect draw buffer of type " + to_string(getResourceType()) + " with " +
                          std::to_string(numBytes) + " bytes. The size of the buffer must be a multiple of " +
                          std::to_string(sizeof(T)));

    GLBaseBuffer::bufferData(data, numBytes, altTarget);
  }

  // void setDataAt(size_t idx, const T& indirectData);

 protected:
  explicit GLBaseIndirectDrawTemplateBuffer(const RendererWkPtr& rendererPtr,
                                            GLResourceType rsrcType,
                                            GLBufferType bufferType,
                                            BufferAccessType accessType,
                                            BufferAccessFreq accessFreq)
      : GLBaseIndirectDrawBuffer(rendererPtr, rsrcType, bufferType, accessType, accessFreq) {}

  explicit GLBaseIndirectDrawTemplateBuffer(const RendererWkPtr& rendererPtr,
                                            GLResourceType rsrcType,
                                            GLBufferType bufferType,
                                            size_t numBytes,
                                            BufferAccessType accessType,
                                            BufferAccessFreq accessFreq)
      : GLBaseIndirectDrawTemplateBuffer(rendererPtr, rsrcType, bufferType, accessType, accessFreq) {
    RUNTIME_EX_ASSERT(numBytes % sizeof(T) == 0,
                      "Cannot allocate an indirect draw buffer of type " + to_string(rsrcType) + " with " +
                          std::to_string(numBytes) + " bytes. The size of the buffer must be a multiple of " +
                          std::to_string(sizeof(T)));

    GLBaseBuffer::bufferData(nullptr, numBytes);
  }

  explicit GLBaseIndirectDrawTemplateBuffer(const RendererWkPtr& rendererPtr,
                                            GLResourceType rsrcType,
                                            GLBufferType bufferType,
                                            const std::vector<T>& items,
                                            BufferAccessType accessType,
                                            BufferAccessFreq accessFreq)
      : GLBaseIndirectDrawTemplateBuffer(rendererPtr, rsrcType, bufferType, accessType, accessFreq) {
    bufferData(items);
  }
};

class GLIndirectDrawVertexBuffer : public GLBaseIndirectDrawTemplateBuffer<IndirectDrawVertexData> {
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

class GLIndirectDrawIndexBuffer : public GLBaseIndirectDrawTemplateBuffer<IndirectDrawIndexData> {
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