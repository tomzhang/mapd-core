#include "GLBufferLayout.h"
#include "GLLayoutBuffer.h"
#include "GLShader.h"

namespace Rendering {
namespace GL {

namespace Resources {

static void initializeAttrTypes(
    std::array<TypeGLUqPtr, static_cast<size_t>(GLBufferAttrType::MAX_GL_BUFFER_ATTR_TYPE)>& attrTypeInfo,
    const std::set<std::string>& ext) {
  attrTypeInfo[static_cast<int>(GLBufferAttrType::UINT)].reset(new TypeGL<unsigned int, 1>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC2UI)].reset(new TypeGL<unsigned int, 2>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC3UI)].reset(new TypeGL<unsigned int, 3>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC4UI)].reset(new TypeGL<unsigned int, 4>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::INT)].reset(new TypeGL<int, 1>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC2I)].reset(new TypeGL<int, 2>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC3I)].reset(new TypeGL<int, 3>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC4I)].reset(new TypeGL<int, 4>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::FLOAT)].reset(new TypeGL<float, 1>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC2F)].reset(new TypeGL<float, 2>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC3F)].reset(new TypeGL<float, 3>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC4F)].reset(new TypeGL<float, 4>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::DOUBLE)].reset(new TypeGL<double, 1>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC2D)].reset(new TypeGL<double, 2>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC3D)].reset(new TypeGL<double, 3>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC4D)].reset(new TypeGL<double, 4>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::UINT64)].reset(new TypeGL<uint64_t, 1>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC2UI64)].reset(new TypeGL<uint64_t, 2>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC3UI64)].reset(new TypeGL<uint64_t, 3>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC4UI64)].reset(new TypeGL<uint64_t, 4>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::INT64)].reset(new TypeGL<int64_t, 1>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC2I64)].reset(new TypeGL<int64_t, 2>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC3I64)].reset(new TypeGL<int64_t, 3>(ext));
  attrTypeInfo[static_cast<int>(GLBufferAttrType::VEC4I64)].reset(new TypeGL<int64_t, 4>(ext));
};

std::array<TypeGLUqPtr, static_cast<size_t>(GLBufferAttrType::MAX_GL_BUFFER_ATTR_TYPE)>
    GLBaseBufferLayout::attrTypeInfo;

GLBufferAttrType GLBaseBufferLayout::getBufferAttrType(unsigned int a, int numComponents) {
  RUNTIME_EX_ASSERT(numComponents == 1, "Only 1 component of unsigned ints are currently supported.");

  return GLBufferAttrType::UINT;
}

GLBufferAttrType GLBaseBufferLayout::getBufferAttrType(int a, int numComponents) {
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

GLBufferAttrType GLBaseBufferLayout::getBufferAttrType(float a, int numComponents) {
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

GLBufferAttrType GLBaseBufferLayout::getBufferAttrType(double a, int numComponents) {
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

GLBufferAttrType GLBaseBufferLayout::getBufferAttrType(uint64_t a, int numComponents) {
  switch (numComponents) {
    case 1:
      return GLBufferAttrType::UINT64;
    case 2:
      return GLBufferAttrType::VEC2UI64;
    case 3:
      return GLBufferAttrType::VEC3UI64;
    case 4:
      return GLBufferAttrType::VEC4UI64;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return GLBufferAttrType::UINT64;
}

GLBufferAttrType GLBaseBufferLayout::getBufferAttrType(int64_t a, int numComponents) {
  switch (numComponents) {
    case 1:
      return GLBufferAttrType::INT64;
    case 2:
      return GLBufferAttrType::VEC2I64;
    case 3:
      return GLBufferAttrType::VEC3I64;
    case 4:
      return GLBufferAttrType::VEC4I64;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(numComponents) + ". Need 1-4 components.");
  }

  return GLBufferAttrType::INT64;
}

GLBaseBufferLayout::GLBaseBufferLayout(const std::set<std::string>& supportedExtensions, GLBufferLayoutType layoutType)
    : _layoutType(layoutType), _itemByteSize(0) {
  if (attrTypeInfo.size() && !attrTypeInfo[0]) {
    initializeAttrTypes(attrTypeInfo, supportedExtensions);
  }
}

bool GLBaseBufferLayout::hasAttribute(const std::string& attrName) const {
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
  return (nameLookup.find(attrName) != nameLookup.end());
}

TypeGLShPtr GLBaseBufferLayout::getAttributeTypeGL(const std::string& attrName) const {
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = nameLookup.find(attrName)) != nameLookup.end(),
                    "GLBaseBufferLayout::getAttributeTypeGL(): attribute " + attrName + " does not exist in layout.");

  return (*itr)->typeInfo->clone();
}

GLBufferAttrType GLBaseBufferLayout::getAttributeType(const std::string& attrName) const {
  // TODO(croot): consolidate this code and the one in getAttributeTypeGL()
  // into a single getBufferAttrInfo func or something.
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = nameLookup.find(attrName)) != nameLookup.end(),
                    "GLBaseBufferLayout::getAttributeType(): attribute " + attrName + " does not exist in layout.");

  return (*itr)->type;
}

const GLBufferAttrInfo& GLBaseBufferLayout::getAttributeInfo(const std::string& attrName) const {
  // TODO(croot): consolidate this code with those in the above two functions
  // into a single getBufferAttrInfo func or something.
  const BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = nameLookup.find(attrName)) != nameLookup.end(),
                    "GLBaseBufferLayout::getAttributeInfo(): attribute " + attrName + " does not exist in layout.");

  return **itr;
}

int GLBaseBufferLayout::getAttributeByteOffset(const std::string& attrName) const {
  return getAttributeInfo(attrName).offset;
}

bool GLBaseBufferLayout::operator==(const GLBaseBufferLayout& layoutPtr) const {
  if (_layoutType != layoutPtr._layoutType) {
    return false;
  }

  if (_attrMap.size() != layoutPtr._attrMap.size()) {
    return false;
  }

  for (size_t i = 0; i < _attrMap.size(); ++i) {
    if (*_attrMap[i] != *layoutPtr._attrMap[i]) {
      return false;
    }
  }
  return true;
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
  _attrMap.push_back(BufferAttrInfoPtr(
      new GLBufferAttrInfo(attrName, type, attrTypeInfo[static_cast<int>(type)].get(), stride, offset)));

  int enumVal = static_cast<int>(type);
  _itemByteSize += attrTypeInfo[enumVal]->numBytes();
}

void GLInterleavedBufferLayout::addAttribute(const std::string& attrName, GLBufferAttrType type) {
  RUNTIME_EX_ASSERT(
      !hasAttribute(attrName) && attrName.length(),
      "GLInterleavedBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

  // TODO(croot), set the stride of all currently existing attrs, or leave
  // that for when the layout is bound to the renderer/shader/VAO

  int enumVal = static_cast<int>(type);
  _attrMap.push_back(
      BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), -1, _itemByteSize)));

  _itemByteSize += attrTypeInfo[enumVal]->numBytes();
}

void GLInterleavedBufferLayout::bindToShader(GLShader* activeShader,
                                             const size_t usedBytes,
                                             const size_t offsetBytes,
                                             const std::string& attr,
                                             const std::string& shaderAttr) const {
  GLuint attrLoc;
  BaseTypeGL* attrPtr;
  GLBufferAttrInfo* bufAttrPtr;

  if (!attr.length()) {
    BufferAttrMap::iterator itr;
    for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
      bufAttrPtr = itr->get();
      attrLoc = activeShader->getVertexAttributeLocation(bufAttrPtr->name);
      attrPtr = bufAttrPtr->typeInfo;
      attrPtr->bind(attrLoc, _itemByteSize, offsetBytes + bufAttrPtr->offset);
    }
  } else {
    const BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
    BufferAttrMap_by_name::iterator itr;

    RUNTIME_EX_ASSERT((itr = nameLookup.find(attr)) != nameLookup.end(),
                      "GLInterleavedBufferLayout::bindToShader(): attribute " + attr + " does not exist in layout.");
    bufAttrPtr = itr->get();
    attrLoc = activeShader->getVertexAttributeLocation(shaderAttr.length() ? shaderAttr : bufAttrPtr->name);
    attrPtr = bufAttrPtr->typeInfo;
    attrPtr->bind(attrLoc, _itemByteSize, offsetBytes + bufAttrPtr->offset);
  }
}

void GLSequentialBufferLayout::addAttribute(const std::string& attrName, GLBufferAttrType type) {
  RUNTIME_EX_ASSERT(
      !hasAttribute(attrName) && attrName.length(),
      "GLSequentialBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

  int enumVal = static_cast<int>(type);
  _attrMap.push_back(BufferAttrInfoPtr(
      new GLBufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), attrTypeInfo[enumVal]->numBytes(), -1)));

  _itemByteSize += attrTypeInfo[enumVal]->numBytes();
}

void GLSequentialBufferLayout::bindToShader(GLShader* activeShader,
                                            const size_t usedBytes,
                                            const size_t offsetBytes,
                                            const std::string& attr,
                                            const std::string& shaderAttr) const {
  GLuint attrLoc;
  BaseTypeGL* attrPtr;
  int offset = offsetBytes;
  auto numVerts = usedBytes / _itemByteSize;

  GLBufferAttrInfo* bufAttrPtr;

  BufferAttrMap::iterator itr;

  if (!attr.length()) {
    for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
      bufAttrPtr = itr->get();
      attrLoc = activeShader->getVertexAttributeLocation(bufAttrPtr->name);
      attrPtr = bufAttrPtr->typeInfo;
      attrPtr->bind(attrLoc, bufAttrPtr->stride, offset);
      MAPD_CHECK_GL_ERROR(glEnableVertexAttribArray(attrLoc));
      offset += attrPtr->numBytes() * numVerts;
    }
  } else {
    RUNTIME_EX_ASSERT(hasAttribute(attr),
                      "GLSequentialBufferLayout::bindToRenderer(): attribute " + attr + " doesn't exist in layout.");

    for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
      bufAttrPtr = itr->get();
      attrPtr = bufAttrPtr->typeInfo;

      if (bufAttrPtr->name == attr) {
        attrLoc = activeShader->getVertexAttributeLocation(shaderAttr.length() ? shaderAttr : bufAttrPtr->name);
        attrPtr->bind(attrLoc, bufAttrPtr->stride, offset);
        MAPD_CHECK_GL_ERROR(glEnableVertexAttribArray(attrLoc));
        break;
      }

      offset += attrPtr->numBytes() * numVerts;
    }
  }
}

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering
