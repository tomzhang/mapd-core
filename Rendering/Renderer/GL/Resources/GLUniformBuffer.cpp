#include "GLUniformBuffer.h"

namespace Rendering {
namespace GL {
namespace Resources {

GLUniformBuffer::GLUniformBuffer(const RendererWkPtr& rendererPtr,
                                 BufferAccessType accessType,
                                 BufferAccessFreq accessFreq)
    : GLBaseBuffer(rendererPtr, GLBufferType::UNIFORM_BUFFER, GL_UNIFORM_BUFFER, accessType, accessFreq),
      _shaderBlockLayoutPtr(nullptr),
      _numItems(0) {
}

// GLUniformBuffer::GLUniformBuffer(const RendererWkPtr& rendererPtr,
//                                  size_t numBytes,
//                                  BufferAccessType accessType,
//                                  BufferAccessFreq accessFreq) : GLUniformBuffer(rendererPtr, accessType, accessFreq)
//                                  {
//   bufferData(nullptr, numBytes, 1);
// }

GLUniformBuffer::GLUniformBuffer(const RendererWkPtr& rendererPtr,
                                 const GLShaderBlockLayoutShPtr& shaderBlockLayoutPtr,
                                 BufferAccessType accessType,
                                 BufferAccessFreq accessFreq)
    : GLUniformBuffer(rendererPtr, accessType, accessFreq) {
  _shaderBlockLayoutPtr = shaderBlockLayoutPtr;
}

GLUniformBuffer::~GLUniformBuffer() {
  cleanupResource();
}

void GLUniformBuffer::_makeEmpty() {
  _shaderBlockLayoutPtr = nullptr;
  _numItems = 0;
}

bool GLUniformBuffer::hasAttribute(const std::string& attrName) const {
  if (!_shaderBlockLayoutPtr) {
    return false;
  }

  return _shaderBlockLayoutPtr->hasAttribute(attrName);
}

TypeGLShPtr GLUniformBuffer::getAttributeTypeGL(const std::string& attrName) const {
  RUNTIME_EX_ASSERT(_shaderBlockLayoutPtr != nullptr,
                    "The uniform buffer has not been properly initialized with a GLShaderBlockLayout object. Cannot "
                    "retrieve the Attribute GL Type of " +
                        attrName + ".");

  return _shaderBlockLayoutPtr->getAttributeTypeGL(attrName);
}

GLBufferAttrType GLUniformBuffer::getAttributeType(const std::string& attrName) const {
  RUNTIME_EX_ASSERT(_shaderBlockLayoutPtr != nullptr,
                    "The uniform buffer has not been properly initialized with a GLShaderBlockLayout. Cannot retrieve "
                    "the Attribute Type of " +
                        attrName + ".");

  return _shaderBlockLayoutPtr->getAttributeType(attrName);
}

int GLUniformBuffer::numAttributes() const {
  return (_shaderBlockLayoutPtr ? _shaderBlockLayoutPtr->numAttributes() : 0);
}

int GLUniformBuffer::numItems() const {
  return _numItems;
}

const GLBufferAttrInfo& GLUniformBuffer::operator[](size_t i) const {
  RUNTIME_EX_ASSERT(_shaderBlockLayoutPtr != nullptr,
                    "The uniform buffer has not been properly initialized with a GLShaderBlockLayout. Cannot retrieve "
                    "attribute at index " +
                        std::to_string(i) + ".");

  return (*_shaderBlockLayoutPtr)[i];
}

void GLUniformBuffer::setBufferLayout(const GLShaderBlockLayoutShPtr& shaderBlockLayoutPtr, size_t numItems) {
}

void GLUniformBuffer::bufferData(void* data, size_t numItems, size_t numBytesPerItem) {
}

// void GLUniformBuffer::_bindToShaderInternal(GLShader* activeShader, const std::string& attr = "", const std::string&
// shaderAttr =
// "");

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
