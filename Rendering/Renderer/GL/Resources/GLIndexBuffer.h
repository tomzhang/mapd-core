#ifndef RENDERING_GL_RESOURCES_GLINDEXBUFFER_H_
#define RENDERING_GL_RESOURCES_GLINDEXBUFFER_H_

#include "Types.h"
#include "GLBaseBuffer.h"
#include "../TypeGL.h"
#include "GLVertexArrayResource.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLIndexBuffer : public GLVertexArrayResource<>, public GLBaseBuffer {
 public:
  enum class IndexType {
    UNSIGNED_BYTE = GL_UNSIGNED_BYTE,
    UNSIGNED_SHORT = GL_UNSIGNED_SHORT,
    UNSIGNED_INT = GL_UNSIGNED_INT
  };

  ~GLIndexBuffer();

  IndexType getIndexType() const { return _indexType; }
  size_t numItems() const { return _numItems; }
  size_t getIndexTypeByteSize() const;

  void bufferData(const void* data, size_t numBytes, GLenum altTarget = 0) final;
  void bufferData(const std::vector<unsigned char>& indices);
  void bufferData(const std::vector<unsigned short>& indices);
  void bufferData(const std::vector<unsigned int>& indices);

 private:
  explicit GLIndexBuffer(const RendererWkPtr& rendererPtr,
                         IndexType indexType = IndexType::UNSIGNED_INT,
                         BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                         BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndexBuffer(const RendererWkPtr& rendererPtr,
                         size_t numBytes,
                         IndexType indexType = IndexType::UNSIGNED_INT,
                         BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                         BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndexBuffer(const RendererWkPtr& rendererPtr,
                         const std::vector<unsigned char>& items,
                         BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                         BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndexBuffer(const RendererWkPtr& rendererPtr,
                         const std::vector<unsigned short>& items,
                         BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                         BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  explicit GLIndexBuffer(const RendererWkPtr& rendererPtr,
                         const std::vector<unsigned int>& items,
                         BufferAccessType accessType = BufferAccessType::READ_AND_WRITE,
                         BufferAccessFreq accessFreq = BufferAccessFreq::STATIC);

  void _makeEmpty() final;

  IndexType _indexType;
  size_t _numItems;

  bool _doesVaoUseThisResource(const GLVertexArrayShPtr& vao) final;
  void _setVaoDirtyFlag(GLVertexArrayShPtr& vao, const bool dirtyFlag) final;

  friend class ::Rendering::GL::GLResourceManager;
};

std::string to_string(const GLIndexBuffer::IndexType value);

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLIndexBuffer::IndexType value);

#endif  // RENDERING_GL_RESOURCES_GLINDEXBUFFER_H_
