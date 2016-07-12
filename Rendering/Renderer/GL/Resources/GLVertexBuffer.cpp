#include "GLVertexBuffer.h"
#include <iostream>

namespace Rendering {
namespace GL {
namespace Resources {

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr,
                               BufferAccessType accessType,
                               BufferAccessFreq accessFreq)
    : GLBaseBuffer(rendererPtr,
                   GLResourceType::VERTEX_BUFFER,
                   GLBufferType::VERTEX_BUFFER,
                   GL_ARRAY_BUFFER,
                   accessType,
                   accessFreq),
      _layoutPtr(nullptr),
      _numItems(0) {
}

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr,
                               size_t numBytes,
                               BufferAccessType accessType,
                               BufferAccessFreq accessFreq)
    : GLVertexBuffer(rendererPtr, accessType, accessFreq) {
  bufferData(nullptr, 1, numBytes);
}

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr,
                               const GLBufferLayoutShPtr& layoutPtr,
                               BufferAccessType accessType,
                               BufferAccessFreq accessFreq)
    : GLVertexBuffer(rendererPtr, accessType, accessFreq) {
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
  size_t bytesPerVertex = layoutPtr->getNumBytesPerItem();

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

  return numBytes() / _layoutPtr->getNumBytesPerItem();
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

void GLVertexBuffer::debugPrintData(void* data, size_t idx) {
  // NOTE: the data ptr should have been populated with a
  // GLBaseBuffer::getBufferData() call.

  RUNTIME_EX_ASSERT(
      _layoutPtr,
      "Cannot print the data for index " + std::to_string(idx) + ". The vbo does not have a buffer layout.");

  unsigned char* chardata = (unsigned char*)(data);

  auto numAttrs = _layoutPtr->numAttributes();
  auto bytesPerVertex = _layoutPtr->getNumBytesPerItem();

  RUNTIME_EX_ASSERT(idx < static_cast<size_t>(numVertices()),
                    "The index " + std::to_string(idx) + " extends beyond the number of vertices in the buffer " +
                        std::to_string(numVertices()));

  auto bufidx = idx / bytesPerVertex;
  std::cout << "[" << idx << "]:" << std::endl;
  for (decltype(numAttrs) i = 0; i < numAttrs; ++i) {
    auto attrinfo = (*_layoutPtr)[i];
    std::cout << "\t[" << i << "] " << attrinfo.name << ": ";
    switch (attrinfo.type) {
      case ::Rendering::GL::Resources::GLBufferAttrType::DOUBLE: {
        double* dataval = (double*)(&chardata[bufidx + attrinfo.offset]);
        std::cout << (*dataval) << std::endl;
      } break;
      case ::Rendering::GL::Resources::GLBufferAttrType::INT: {
        bool doInt64 = false;
        if (i < numAttrs - 1) {
          auto nextattrinfo = (*_layoutPtr)[i + 1];
          if (nextattrinfo.name.find("_dummy") == 0) {
            doInt64 = true;
          }
        }

        if (doInt64) {
          int64_t* dataval = (int64_t*)(&chardata[bufidx + attrinfo.offset]);
          std::cout << (*dataval) << std::endl;
          ++i;
        } else {
          int* dataval = (int*)(&chardata[bufidx + attrinfo.offset]);
          std::cout << (*dataval) << std::endl;
        }
      } break;
      default:
        std::cout << "- currently unsupported type: " << static_cast<int>(attrinfo.type) << std::endl;
        break;
    }
  }
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
