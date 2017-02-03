#include "GLIndexBuffer.h"

namespace Rendering {
namespace GL {
namespace Resources {

std::string to_string(const GLIndexBuffer::IndexType value) {
  switch (value) {
    case GLIndexBuffer::IndexType::UNSIGNED_BYTE:
      return "UNSIGNED_BYTE";
    case GLIndexBuffer::IndexType::UNSIGNED_SHORT:
      return "UNSIGNED_SHORT";
    case GLIndexBuffer::IndexType::UNSIGNED_INT:
      return "UNSIGNED_INT";
    default:
      CHECK(false) << "Unsupported index type " << static_cast<int>(value) << ". Cannot convert it to a string.";
  }

  return "";
}

static size_t getIndexTypeSize(GLIndexBuffer::IndexType indexType) {
  switch (indexType) {
    case GLIndexBuffer::IndexType::UNSIGNED_BYTE:
      return sizeof(unsigned char);
    case GLIndexBuffer::IndexType::UNSIGNED_SHORT:
      return sizeof(unsigned short);
    case GLIndexBuffer::IndexType::UNSIGNED_INT:
      return sizeof(unsigned int);
    default:
      CHECK(false) << "Unsupported index type " << static_cast<int>(indexType) << ". Cannot get its size.";
  }
  return 0;
}

GLIndexBuffer::GLIndexBuffer(const RendererWkPtr& rendererPtr,
                             IndexType indexType,
                             BufferAccessType accessType,
                             BufferAccessFreq accessFreq)
    : GLBaseBuffer(rendererPtr,
                   GLResourceType::INDEX_BUFFER,
                   GLBufferType::INDEX_BUFFER,
                   GL_ELEMENT_ARRAY_BUFFER,
                   accessType,
                   accessFreq),
      _indexType(indexType),
      _numItems(0) {}

GLIndexBuffer::GLIndexBuffer(const RendererWkPtr& rendererPtr,
                             size_t numBytes,
                             IndexType indexType,
                             BufferAccessType accessType,
                             BufferAccessFreq accessFreq)
    : GLIndexBuffer(rendererPtr, indexType, accessType, accessFreq) {
  bufferData(nullptr, numBytes);
}

GLIndexBuffer::GLIndexBuffer(const RendererWkPtr& rendererPtr,
                             const std::vector<unsigned char>& items,
                             BufferAccessType accessType,
                             BufferAccessFreq accessFreq)
    : GLIndexBuffer(rendererPtr, IndexType::UNSIGNED_BYTE, accessType, accessFreq) {
  bufferData(items);
}

GLIndexBuffer::GLIndexBuffer(const RendererWkPtr& rendererPtr,
                             const std::vector<unsigned short>& items,
                             BufferAccessType accessType,
                             BufferAccessFreq accessFreq)
    : GLIndexBuffer(rendererPtr, IndexType::UNSIGNED_SHORT, accessType, accessFreq) {
  bufferData(items);
}

GLIndexBuffer::GLIndexBuffer(const RendererWkPtr& rendererPtr,
                             const std::vector<unsigned int>& items,
                             BufferAccessType accessType,
                             BufferAccessFreq accessFreq)
    : GLIndexBuffer(rendererPtr, IndexType::UNSIGNED_INT, accessType, accessFreq) {
  bufferData(items);
}

GLIndexBuffer::~GLIndexBuffer() {
  cleanupResource();
}

void GLIndexBuffer::_makeEmpty() {
  _numItems = 0;
  GLVertexArrayResource::_cleanupResource();
}

size_t GLIndexBuffer::getIndexTypeByteSize() const {
  return getIndexTypeSize(_indexType);
}

void GLIndexBuffer::bufferData(const void* data, size_t numBytes, GLenum altTarget) {
  auto typesz = getIndexTypeByteSize();
  RUNTIME_EX_ASSERT(numBytes % typesz == 0,
                    "Cannot allocate an index buffer of type " + to_string(_indexType) + " with " +
                        std::to_string(numBytes) + " bytes. The size of the buffer must be a multiple of " +
                        std::to_string(typesz));

  RUNTIME_EX_ASSERT(_vaoSize() == 0,
                    "There are existing vertex array objects that are currently making use of the index buffer. "
                    "Cannot buffer a full set of new data as it would invalidate those vaos. Delete/clear those vaos "
                    "first.");

  GLBaseBuffer::bufferData(data, numBytes, altTarget);

  _numItems = numBytes / typesz;
}

void GLIndexBuffer::bufferData(const std::vector<unsigned char>& indices) {
  RUNTIME_EX_ASSERT(_indexType == IndexType::UNSIGNED_BYTE,
                    "Cannot set data of type " + to_string(IndexType::UNSIGNED_BYTE) +
                        " for an index buffer of type: " + to_string(_indexType) + ".");
  RUNTIME_EX_ASSERT(_vaoSize() == 0,
                    "There are existing vertex array objects that are currently making use of the index buffer. "
                    "Cannot buffer a full set of new data as it would invalidate those vaos. Delete/clear those vaos "
                    "first.");
  GLBaseBuffer::bufferData(&indices[0], indices.size() * sizeof(unsigned char));
  _numItems = indices.size();
}

void GLIndexBuffer::bufferData(const std::vector<unsigned short>& indices) {
  RUNTIME_EX_ASSERT(_indexType == IndexType::UNSIGNED_SHORT,
                    "Cannot set data of type " + to_string(IndexType::UNSIGNED_SHORT) +
                        " for an index buffer of type: " + to_string(_indexType) + ".");
  RUNTIME_EX_ASSERT(_vaoSize() == 0,
                    "There are existing vertex array objects that are currently making use of the index buffer. "
                    "Cannot buffer a full set of new data as it would invalidate those vaos. Delete/clear those vaos "
                    "first.");
  GLBaseBuffer::bufferData(&indices[0], indices.size() * sizeof(unsigned short));
  _numItems = indices.size();
}

void GLIndexBuffer::bufferData(const std::vector<unsigned int>& indices) {
  RUNTIME_EX_ASSERT(_indexType == IndexType::UNSIGNED_INT,
                    "Cannot set data of type " + to_string(IndexType::UNSIGNED_INT) + " for an index buffer of type: " +
                        to_string(_indexType) + ".");
  RUNTIME_EX_ASSERT(_vaoSize() == 0,
                    "There are existing vertex array objects that are currently making use of the index buffer. "
                    "Cannot buffer a full set of new data as it would invalidate those vaos. Delete/clear those vaos "
                    "first.");
  GLBaseBuffer::bufferData(&indices[0], indices.size() * sizeof(unsigned int));
  _numItems = indices.size();
}

bool GLIndexBuffer::_doesVaoUseThisResource(const GLVertexArrayShPtr& vao) {
  CHECK(vao);
  return vao->hasIbo(this);
}

void GLIndexBuffer::_setVaoDirtyFlag(GLVertexArrayShPtr& vao, const bool dirtyFlag) {
  vao->_setDirtyFlag(dirtyFlag);
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLIndexBuffer::IndexType value) {
  os << ::Rendering::GL::Resources::to_string(value);
  return os;
}
