#include "GLVertexBuffer.h"
#include "../GLResourceManager.h"

// #include <boost/lambda/lambda.hpp>
#include <iostream>
#include <algorithm>

namespace Rendering {
namespace GL {
namespace Resources {

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr,
                               BufferAccessType accessType,
                               BufferAccessFreq accessFreq)
    : GLLayoutBuffer<GLBaseBufferLayout>(rendererPtr,
                                         GLResourceType::VERTEX_BUFFER,
                                         GLBufferType::VERTEX_BUFFER,
                                         GL_ARRAY_BUFFER,
                                         accessType,
                                         accessFreq) {
}

GLVertexBuffer::GLVertexBuffer(const RendererWkPtr& rendererPtr,
                               size_t numBytes,
                               BufferAccessType accessType,
                               BufferAccessFreq accessFreq)
    : GLVertexBuffer(rendererPtr, accessType, accessFreq) {
  bufferData(nullptr, numBytes);
}

GLVertexBuffer::~GLVertexBuffer() {
  cleanupResource();
}

void GLVertexBuffer::_makeEmpty() {
  GLLayoutManagerBuffer::_makeEmpty();
  _deleteAllVertexArrays();
}

size_t GLVertexBuffer::numVertices(const GLBufferLayoutShPtr& layoutPtr) const {
  return numItems(layoutPtr);
}

void GLVertexBuffer::bufferData(void* data, const size_t numBytes, const GLBufferLayoutShPtr& layoutPtr) {
  // if there are any vertex array objects using this vbo, they
  // would all be invalidated by this call. For now, throw an error.
  // TODO(croot): instead of throwing an error, should we invalidate
  // the vao and throw an error at the point it is attempted to be used
  // again (i.e. vao->setUsable(false))?

  _cleanupVaoRefs();
  RUNTIME_EX_ASSERT(!_vaoRefs.size() || (_vaoRefs.size() == 1 && _vaoRefs.begin()->lock()->hasVbo(this, layoutPtr)),
                    "There are existing vertex array objects that are currently making use of the vertex buffer. "
                    "Cannot buffer a full set of new data as it would invalidate those vaos. Delete/clear those vaos "
                    "first.");

  GLLayoutBuffer<GLBaseBufferLayout>::bufferData(data, numBytes, layoutPtr);
}

void GLVertexBuffer::_bindToShaderInternal(GLShader* activeShader,
                                           const GLBufferLayoutShPtr& layoutPtr,
                                           const std::string& attr,
                                           const std::string& shaderAttr) {
  RUNTIME_EX_ASSERT(_bufferId != 0, "Cannot bind vertex buffer. It has not been initialized with data.");
  auto layoutData = _getBufferLayoutDataToUse(layoutPtr, "Cannot bind vertex buffer to shader. ");

  MAPD_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, _bufferId));
  layoutData.layoutPtr->bindToShader(activeShader, layoutData.usedBytes, layoutData.offsetBytes, attr, shaderAttr);
}

void GLVertexBuffer::debugPrintData(void* data, const GLBufferLayoutShPtr& layoutPtr, size_t idx) {
  // NOTE: the data ptr should have been populated with a
  // GLBaseBuffer::getBufferData() call.

  // TODO(croot): no real need to have the user supply the data argument.
  // We can gather the data here using the getBufferData call
  // since we have the layout and know the offset and size to gether.
  // The only issue is repeated calls to this function. Perhaps a better
  // way is to supply a list of idxs to debug or somehow provide an
  // iterator to iterate through.

  RUNTIME_EX_ASSERT(layoutPtr, "A layout is needed to properly print data");

  auto layoutData = getBufferLayoutData(layoutPtr);
  auto offset = layoutData.second;

  unsigned char* chardata = (unsigned char*)(data);

  auto numAttrs = layoutPtr->numAttributes();
  auto bytesPerVertex = layoutPtr->getNumBytesPerItem();

  RUNTIME_EX_ASSERT(idx < static_cast<size_t>(numVertices(layoutPtr)),
                    "The index " + std::to_string(idx) +
                        " extends beyond the number of vertices in the layout-defined buffer " +
                        std::to_string(numVertices(layoutPtr)));

  auto bufidx = offset + idx * bytesPerVertex;
  std::cout << "[" << idx << "]:" << std::endl;
  for (decltype(numAttrs) i = 0; i < numAttrs; ++i) {
    auto attrinfo = (*layoutPtr)[i];
    std::cout << "\t[" << i << "] " << attrinfo.name << ": ";
    switch (attrinfo.type) {
      case ::Rendering::GL::Resources::GLBufferAttrType::DOUBLE: {
        double* dataval = (double*)(&chardata[bufidx + attrinfo.offset]);
        std::cout << (*dataval) << std::endl;
      } break;
      case ::Rendering::GL::Resources::GLBufferAttrType::INT: {
        bool doInt64 = false;
        if (i < numAttrs - 1) {
          auto nextattrinfo = (*layoutPtr)[i + 1];
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
  _cleanupVaoRefs();

  auto vaoRsrc = getSharedResourceFromTypePtr(vao);
  if (vaoRsrc) {
    _vaoRefs.erase(vaoRsrc);
  }
}

void GLVertexBuffer::_deleteAllVertexArrays() {
  GLVertexArrayShPtr vaoPtr;

  for (auto& vao : _vaoRefs) {
    vaoPtr = vao.lock();
    if (vaoPtr) {
      // vaoPtr->_deleteVertexBuffer(this);
      vaoPtr->_setDirtyFlag(true);
    }
  }

  _vaoRefs.clear();
}

void GLVertexBuffer::_markVertexArraysDirty(const GLBufferLayoutShPtr& layoutPtr) {
  GLVertexArrayShPtr vaoPtr;
  std::vector<GLVertexArrayWkPtr> vaosToDelete;
  for (auto& vao : _vaoRefs) {
    vaoPtr = vao.lock();
    if (vaoPtr && vaoPtr->hasVbo(this, layoutPtr)) {
      vaoPtr->_setDirtyFlag(true);
    } else {
      vaosToDelete.push_back(vao);
    }
  }

  for (auto& vaoWkPtr : vaosToDelete) {
    _vaoRefs.erase(vaoWkPtr);
  }
}

void GLVertexBuffer::_cleanupVaoRefs() {
  GLVertexArrayShPtr vaoPtr;
  std::vector<GLVertexArrayWkPtr> vaosToDelete;
  for (auto& vaoWkPtr : _vaoRefs) {
    if (vaoWkPtr.expired()) {
      vaosToDelete.push_back(vaoWkPtr);
    }
  }

  for (auto& vaoWkPtr : vaosToDelete) {
    _vaoRefs.erase(vaoWkPtr);
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
