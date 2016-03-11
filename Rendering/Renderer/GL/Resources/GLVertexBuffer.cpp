#include "GLVertexBuffer.h"

namespace Rendering {
namespace GL {
namespace Resources {

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr, GLenum usage)
    : GLBaseBuffer(rendererPtr, GLBufferType::VERTEX_BUFFER, GL_ARRAY_BUFFER, usage),
      _layoutPtr(nullptr),
      _numItems(0) {
}

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr, size_t numBytes, GLenum usage)
    : GLVertexBuffer(rendererPtr, usage) {
  bufferData(nullptr, numBytes, 1);
}

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr, const GLBufferLayoutShPtr& layoutPtr, GLenum usage)
    : GLVertexBuffer(rendererPtr, usage) {
  _layoutPtr = layoutPtr;
}

GLVertexBuffer::~GLVertexBuffer() {
  cleanupResource();
}

void GLVertexBuffer::_makeEmpty() {
  GLBaseBuffer::_makeEmpty();
  _layoutPtr = nullptr;
  _numItems = 0;
  _deleteAllVertexArrays();
}

bool GLVertexBuffer::hasAttribute(const std::string& attrName) const {
  if (!_layoutPtr) {
    return false;
  }
  return _layoutPtr->hasAttribute(attrName);
}

TypeGLShPtr GLVertexBuffer::getAttributeTypeGL(const std::string& attrName) const {
  RUNTIME_EX_ASSERT(_layoutPtr != nullptr,
                    "The vertex buffer has not been properly initialized. Cannot retrieve the Attribute GL Type of " +
                        attrName + ".");

  return _layoutPtr->getAttributeTypeGL(attrName);
}

GLBufferAttrType GLVertexBuffer::getAttributeType(const std::string& attrName) const {
  RUNTIME_EX_ASSERT(
      _layoutPtr != nullptr,
      "The vertex buffer has not been properly initialized. Cannot retrieve the Attribute Type of " + attrName + ".");

  return _layoutPtr->getAttributeType(attrName);
}

void GLVertexBuffer::setBufferLayout(const GLBufferLayoutShPtr& layoutPtr, size_t numItems) {
  size_t bytesPerVertex = layoutPtr->getNumBytesPerVertex();

  if (bytesPerVertex * numItems > numBytes()) {
    LOG(WARNING) << "Total number of bytes in vertex buffer after setting a new buffer layout exceeds the current size "
                    "of the buffer. Consider allocating the appropriate size for the buffer beforehand. This may "
                    "result in undefined behavior when rendering.";
  }

  _layoutPtr = layoutPtr;
  _numItems = numItems;

  // notify any VAOs this buffer is attached to about its change in size
  _updateVertexArrays();
}

int GLVertexBuffer::numAttributes() const {
  return (_layoutPtr ? _layoutPtr->numAttributes() : 0);
}

int GLVertexBuffer::numItems() const {
  return _numItems;
}

int GLVertexBuffer::numVertices() const {
  RUNTIME_EX_ASSERT(_layoutPtr != nullptr,
                    "The vertex buffer has not been properly initialized with a layout. Cannot retrieve the number of "
                    "vertices in the buffer.");

  return numBytes() / _layoutPtr->getNumBytesPerVertex();
}

const GLBufferAttrInfo& GLVertexBuffer::operator[](size_t i) const {
  RUNTIME_EX_ASSERT(_layoutPtr != nullptr,
                    "The vertex buffer has not been properly initialized. Cannot retrieve attribute at index " +
                        std::to_string(i) + ".");

  return (*_layoutPtr)[i];
}

void GLVertexBuffer::bindToShader(GLShader* activeShader, const std::string& attr, const std::string& shaderAttr) {
  RUNTIME_EX_ASSERT(_bufferId != 0, "Cannot bind vertex buffer. It has not been initialized with data.");
  RUNTIME_EX_ASSERT(_layoutPtr != nullptr, "Cannot bind vertex buffer. It does not have a defined layout.");

  GLint currVbo;
  MAPD_CHECK_GL_ERROR(glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currVbo));

  // TODO(croot): what if the following call throws and error but it is caught appropriately?
  _bindToShaderInternal(activeShader, attr, shaderAttr);

  if (currVbo != static_cast<GLint>(_bufferId)) {
    MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, currVbo));
  }
}

void GLVertexBuffer::_bindToShaderInternal(GLShader* activeShader,
                                           const std::string& attr,
                                           const std::string& shaderAttr) {
  RUNTIME_EX_ASSERT(_bufferId != 0, "Cannot bind vertex buffer. It has not been initialized with data.");
  RUNTIME_EX_ASSERT(_layoutPtr != nullptr, "Cannot bind vertex buffer. It does not have a defined layout.");

  MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, _bufferId));
  _layoutPtr->bindToShader(activeShader, numItems(), attr, shaderAttr);
}

void GLVertexBuffer::bufferData(void* data, size_t numItems, size_t numBytesPerItem) {
  GLBaseBuffer::bufferData(data, numItems * numBytesPerItem);
  _numItems = numItems;

  // notify all vaos this buffer is attached to about the change in buffer size.
  _updateVertexArrays();
}

void GLVertexBuffer::_addVertexArray(GLVertexArrayShPtr& vao) {
  _vaoRefs.emplace(vao);
}

void GLVertexBuffer::_deleteVertexArray(GLVertexArray* vao) {
  GLVertexArrayShPtr vaoPtr;
  std::vector<GLVertexArrayWkPtr> vaosToDelete;
  for (auto& vaoWkPtr : _vaoRefs) {
    vaoPtr = vaoWkPtr.lock();
    if (vaoPtr && vaoPtr.get() == vao) {
      vaosToDelete.push_back(vaoWkPtr);
    } else if (!vaoPtr) {
      vaosToDelete.push_back(vaoWkPtr);
    }
  }

  for (auto& vaoWkPtr : vaosToDelete) {
    _vaoRefs.erase(vaoWkPtr);
  }
}

void GLVertexBuffer::_deleteAllVertexArrays() {
  GLVertexArrayShPtr vaoPtr;
  for (auto& vao : _vaoRefs) {
    vaoPtr = vao.lock();
    if (vaoPtr) {
      vaoPtr->_deleteVertexBuffer(this);
    }
  }

  _vaoRefs.clear();
}

void GLVertexBuffer::_updateVertexArrays() {
  GLVertexArrayShPtr vaoPtr;
  for (auto& vao : _vaoRefs) {
    vaoPtr = vao.lock();
    if (vaoPtr) {
      vaoPtr->_vboUpdated(this);
    }
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
