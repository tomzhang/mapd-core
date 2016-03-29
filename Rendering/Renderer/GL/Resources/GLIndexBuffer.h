#ifndef RENDERING_GL_RESOURCES_GLINDEXBUFFER_H_
#define RENDERING_GL_RESOURCES_GLINDEXBUFFER_H_

#include "Types.h"
#include "GLBaseBuffer.h"
#include "../TypeGL.h"

namespace Rendering {
namespace GL {
namespace Resources {

class GLIndexBuffer : public GLBaseBuffer {
 public:
  enum class IndexType {
    UNSIGNED_BYTE = GL_UNSIGNED_BYTE,
    UNSIGNED_SHORT = GL_UNSIGNED_SHORT,
    UNSIGNED_INT = GL_UNSIGNED_INT
  };

  ~GLIndexBuffer();

  GLResourceType getResourceType() const final { return GLResourceType::INDEXBUFFER; }

  IndexType getIndexType() const { return _indexType; }
  size_t numItems() const { return _numItems; }
  size_t getIndexTypeByteSize() const;

  void bufferData(const std::vector<unsigned char>& indices);
  void bufferData(const std::vector<unsigned short>& indices);
  void bufferData(const std::vector<unsigned int>& indices);

  // template <typename T>
  // void setAttrData(size_t idx, const std::string& attrName, T attrValue) {
  //   _bufferSubData(attrName, &attrValue, sizeof(T), 1, idx);
  // }

  // template <typename T>
  // void setAttrData(size_t idx, const std::string& attrName, const std::vector<T>& attrValue) {
  //   _bufferSubData(attrName, &attrValue[0], sizeof(T), attrValue.size(), idx);
  // }

  // template <typename T, size_t N>
  // void setAttrData(size_t idx, const std::string& attrName, const std::array<T, N>& attrValue) {
  //   _bufferSubData(attrName, &attrValue[0], sizeof(T), N, idx);
  // }

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

  // void _bindToShaderInternal(GLShader* activeShader, const std::string& attr = "", const std::string& shaderAttr =
  // "");

  IndexType _indexType;
  size_t _numItems;

  friend class ::Rendering::GL::GLResourceManager;
};

std::string to_string(const GLIndexBuffer::IndexType value);

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLIndexBuffer::IndexType value);

#endif  // RENDERING_GL_RESOURCES_GLINDEXBUFFER_H_
