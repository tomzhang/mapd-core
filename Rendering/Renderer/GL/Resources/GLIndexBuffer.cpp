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
    : GLBaseBuffer(rendererPtr, GLBufferType::INDEX_BUFFER, GL_ELEMENT_ARRAY_BUFFER, accessType, accessFreq),
      _indexType(indexType),
      _numItems(0) {
}

GLIndexBuffer::GLIndexBuffer(const RendererWkPtr& rendererPtr,
                             size_t numBytes,
                             IndexType indexType,
                             BufferAccessType accessType,
                             BufferAccessFreq accessFreq)
    : GLIndexBuffer(rendererPtr, indexType, accessType, accessFreq) {
  RUNTIME_EX_ASSERT(numBytes % getIndexTypeSize(indexType) == 0,
                    "Cannot allocate an index buffer of type " + to_string(indexType) + " with " +
                        std::to_string(numBytes) + " bytes. The size of the buffer must be a multiple of " +
                        std::to_string(getIndexTypeSize(indexType)));

  GLBaseBuffer::bufferData(nullptr, numBytes);
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
}

void GLIndexBuffer::bufferData(const std::vector<unsigned char>& indices) {
  RUNTIME_EX_ASSERT(_indexType == IndexType::UNSIGNED_BYTE,
                    "Cannot set data of type " + to_string(IndexType::UNSIGNED_BYTE) +
                        " for an index buffer of type: " + to_string(_indexType) + ".");
  GLBaseBuffer::bufferData(&indices[0], indices.size() * sizeof(unsigned char));
  _numItems = indices.size();
}

void GLIndexBuffer::bufferData(const std::vector<unsigned short>& indices) {
  RUNTIME_EX_ASSERT(_indexType == IndexType::UNSIGNED_SHORT,
                    "Cannot set data of type " + to_string(IndexType::UNSIGNED_SHORT) +
                        " for an index buffer of type: " + to_string(_indexType) + ".");
  GLBaseBuffer::bufferData(&indices[0], indices.size() * sizeof(unsigned short));
  _numItems = indices.size();
}

void GLIndexBuffer::bufferData(const std::vector<unsigned int>& indices) {
  RUNTIME_EX_ASSERT(_indexType == IndexType::UNSIGNED_INT,
                    "Cannot set data of type " + to_string(IndexType::UNSIGNED_INT) + " for an index buffer of type: " +
                        to_string(_indexType) + ".");
  GLBaseBuffer::bufferData(&indices[0], indices.size() * sizeof(unsigned int));
  _numItems = indices.size();
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

std::ostream& operator<<(std::ostream& os, const Rendering::GL::Resources::GLIndexBuffer::IndexType value) {
  os << ::Rendering::GL::Resources::to_string(value);
  return os;
}
