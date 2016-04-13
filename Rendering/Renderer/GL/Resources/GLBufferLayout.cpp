#include "GLBufferLayout.h"
#include "GLShader.h"

namespace Rendering {
namespace GL {

namespace Resources {

namespace detail {

GLBufferAttrType getBufferAttrType(unsigned int a, int numComponents) {
  RUNTIME_EX_ASSERT(numComponents == 1, "Only 1 component of unsigned ints are currently supported.");

  return GLBufferAttrType::UINT;
}

GLBufferAttrType getBufferAttrType(int a, int numComponents) {
  switch (numComponents) {
    case 1:
      return GLBufferAttrType::INT;
    case 2:
      return GLBufferAttrType::VEC2I;
    case 3:
      return GLBufferAttrType::VEC3I;
    case 4:
      return GLBufferAttrType::VEC4I;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return GLBufferAttrType::INT;
}

GLBufferAttrType getBufferAttrType(float a, int numComponents) {
  switch (numComponents) {
    case 1:
      return GLBufferAttrType::FLOAT;
    case 2:
      return GLBufferAttrType::VEC2F;
    case 3:
      return GLBufferAttrType::VEC3F;
    case 4:
      return GLBufferAttrType::VEC4F;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return GLBufferAttrType::FLOAT;
}

GLBufferAttrType getBufferAttrType(double a, int numComponents) {
  switch (numComponents) {
    case 1:
      return GLBufferAttrType::DOUBLE;
    case 2:
      return GLBufferAttrType::VEC2D;
    case 3:
      return GLBufferAttrType::VEC3D;
    case 4:
      return GLBufferAttrType::VEC4D;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return GLBufferAttrType::DOUBLE;
}

}  // namespace detail

bool GLBaseBufferLayout::hasAttribute(const std::string& attrName) const {
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<MultiIndexTags::name>();
  return (nameLookup.find(attrName) != nameLookup.end());
  // return (_attrMap.find(attrName) != _attrMap.end());
}

TypeGLShPtr GLBaseBufferLayout::getAttributeTypeGL(const std::string& attrName) const {
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<MultiIndexTags::name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = nameLookup.find(attrName)) != nameLookup.end(),
                    "GLBaseBufferLayout::getAttributeTypeGL(): attribute " + attrName + " does not exist in layout.");

  return (*itr)->typeInfo->clone();
}

GLBufferAttrType GLBaseBufferLayout::getAttributeType(const std::string& attrName) const {
  // TODO(croot): consolidate this code and the one in getAttributeTypeGL()
  // into a single getBufferAttrInfo func or something.
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<MultiIndexTags::name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = nameLookup.find(attrName)) != nameLookup.end(),
                    "GLBaseBufferLayout::getAttributeType(): attribute " + attrName + " does not exist in layout.");

  return (*itr)->type;
}

const GLBufferAttrInfo& GLBaseBufferLayout::getAttributeInfo(const std::string& attrName) const {
  // TODO(croot): consolidate this code with those in the above two functions
  // into a single getBufferAttrInfo func or something.
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<MultiIndexTags::name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = nameLookup.find(attrName)) != nameLookup.end(),
                    "GLBaseBufferLayout::getAttributeInfo(): attribute " + attrName + " does not exist in layout.");

  return **itr;
}

int GLBaseBufferLayout::getAttributeByteOffset(const std::string& attrName) const {
  return getAttributeInfo(attrName).offset;
}

const GLBufferAttrInfo& GLBaseBufferLayout::operator[](size_t i) const {
  RUNTIME_EX_ASSERT(i < _attrMap.size(),
                    "GLBaseBufferLayout::operator[]: cannot retrieve attribute info at index: " + std::to_string(i) +
                        ". The layout only has " + std::to_string(_attrMap.size()) + " attributes.");

  return *_attrMap[i];
}

void GLCustomBufferLayout::addAttribute(const std::string& attrName, GLBufferAttrType type, int stride, int offset) {
  RUNTIME_EX_ASSERT(!hasAttribute(attrName) && attrName.length(),
                    "GLCustomBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

  // _attrMap[attrName] = BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type,
  // detail::attrTypeInfo[static_cast<int>(type)].get(), stride, offset));

  // _attrMap.emplace_back(new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[static_cast<int>(type)].get(),
  // stride,
  // offset));
  _attrMap.push_back(BufferAttrInfoPtr(
      new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[static_cast<int>(type)].get(), stride, offset)));

  int enumVal = static_cast<int>(type);
  _vertexByteSize += detail::attrTypeInfo[enumVal]->numBytes();
}

void GLInterleavedBufferLayout::addAttribute(const std::string& attrName, GLBufferAttrType type) {
  RUNTIME_EX_ASSERT(
      !hasAttribute(attrName) && attrName.length(),
      "GLInterleavedBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

  // TODO, set the stride of all currently existing attrs, or leave
  // that for when the layout is bound to the renderer/shader/VAO

  int enumVal = static_cast<int>(type);
  // _attrMap[attrName] = BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[enumVal].get(),
  // -1,
  // _vertexByteSize));

  // _attrMap.emplace_back(new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[enumVal].get(), -1,
  // _vertexByteSize));
  _attrMap.push_back(BufferAttrInfoPtr(
      new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[enumVal].get(), -1, _vertexByteSize)));

  _vertexByteSize += detail::attrTypeInfo[enumVal]->numBytes();
}

void GLInterleavedBufferLayout::bindToShader(GLShader* activeShader,
                                             int numActiveBufferItems,
                                             const std::string& attr,
                                             const std::string& shaderAttr) {
  GLuint attrLoc;
  BaseTypeGL* attrPtr;

  GLBufferAttrInfo* bufAttrPtr;

  if (!attr.length()) {
    BufferAttrMap::iterator itr;
    for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
      bufAttrPtr = itr->get();
      attrLoc = activeShader->getVertexAttributeLocation(bufAttrPtr->name);
      attrPtr = bufAttrPtr->typeInfo;
      attrPtr->bind(attrLoc, _vertexByteSize, bufAttrPtr->offset);
    }
  } else {
    BufferAttrMap_by_name& nameLookup = _attrMap.get<MultiIndexTags::name>();
    BufferAttrMap_by_name::iterator itr;

    RUNTIME_EX_ASSERT((itr = nameLookup.find(attr)) != nameLookup.end(),
                      "GLInterleavedBufferLayout::bindToShader(): attribute " + attr + " does not exist in layout.");
    bufAttrPtr = itr->get();
    attrLoc = activeShader->getVertexAttributeLocation(shaderAttr.length() ? shaderAttr : bufAttrPtr->name);
    attrPtr = bufAttrPtr->typeInfo;
    attrPtr->bind(attrLoc, _vertexByteSize, bufAttrPtr->offset);
  }
}

void GLSequentialBufferLayout::addAttribute(const std::string& attrName, GLBufferAttrType type) {
  RUNTIME_EX_ASSERT(
      !hasAttribute(attrName) && attrName.length(),
      "GLSequentialBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

  int enumVal = static_cast<int>(type);
  // _attrMap[attrName] = BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[enumVal].get(),
  // detail::attrTypeInfo[enumVal]->numBytes(), -1));
  // _attrMap.emplace_back(new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[enumVal].get(),
  // detail::attrTypeInfo[enumVal]->numBytes(), -1));
  _attrMap.push_back(BufferAttrInfoPtr(new GLBufferAttrInfo(
      attrName, type, detail::attrTypeInfo[enumVal].get(), detail::attrTypeInfo[enumVal]->numBytes(), -1)));

  _vertexByteSize += detail::attrTypeInfo[enumVal]->numBytes();
}

void GLSequentialBufferLayout::bindToShader(GLShader* activeShader,
                                            int numActiveBufferItems,
                                            const std::string& attr,
                                            const std::string& shaderAttr) {
  GLuint attrLoc;
  BaseTypeGL* attrPtr;
  int offset = 0;

  GLBufferAttrInfo* bufAttrPtr;

  BufferAttrMap::iterator itr;

  if (!attr.length()) {
    for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
      bufAttrPtr = itr->get();
      attrLoc = activeShader->getVertexAttributeLocation(bufAttrPtr->name);
      attrPtr = bufAttrPtr->typeInfo;
      // glVertexAttribPointer(attrLoc, attrPtr->numComponents(), attrPtr->baseGLType(), GL_FALSE, bufAttrPtr->stride,
      // BUFFER_OFFFSET(offset));
      attrPtr->bind(attrLoc, bufAttrPtr->stride, offset);
      MAPD_CHECK_GL_ERROR(glEnableVertexAttribArray(attrLoc));
      offset += attrPtr->numBytes() * numActiveBufferItems;
    }
  } else {
    RUNTIME_EX_ASSERT(hasAttribute(attr),
                      "GLSequentialBufferLayout::bindToRenderer(): attribute " + attr + " doesn't exist in layout.");

    for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
      bufAttrPtr = itr->get();
      attrPtr = bufAttrPtr->typeInfo;

      if (bufAttrPtr->name == attr) {
        attrLoc = activeShader->getVertexAttributeLocation(shaderAttr.length() ? shaderAttr : bufAttrPtr->name);
        // MAPD_CHECK_GL_ERROR(glVertexAttribPointer(attrLoc, attrPtr->numComponents(), attrPtr->baseGLType(),
        // GL_FALSE,
        // bufAttrPtr->stride, BUFFER_OFFSET(offset)));
        attrPtr->bind(attrLoc, bufAttrPtr->stride, offset);
        MAPD_CHECK_GL_ERROR(glEnableVertexAttribArray(attrLoc));
        break;
      }

      offset += attrPtr->numBytes() * numActiveBufferItems;
    }
  }

  // for (const auto& itr : _attrMap) {
  //     attrLoc = activeShader->getVertexAttributeLocation(itr.first);
  //     attrPtr = itr.second->typeInfo;
  //     MAPD_CHECK_GL_ERROR(glVertexAttribPointer(attrLoc, attrPtr->numComponents(), attrPtr->baseGLType(), GL_FALSE,
  //     itr.second->stride,
  //     BUFFER_OFFSET(offset)));
  //     MAPD_CHECK_GL_ERROR(glEnableVertexAttribArray(attrLoc));
  //     offset += attrPtr->numBytes() * numActiveBufferItems;
  // }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
