#ifndef BUFFER_LAYOUT_H_
#define BUFFER_LAYOUT_H_

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>

#include <array>
#include <vector>
#include <string>
#include <assert.h>
#include <utility>  // std::pair, std::make_pair
#include <memory>   // std::unique_ptr
#include <GL/glew.h>
#include "TypeGL.h"
#include "Shader.h"
#include <cstdint>

using namespace ::boost;
using namespace ::boost::multi_index;

namespace MapD_Renderer {

enum LayoutType { INTERLEAVED = 0, SEQUENTIAL, CUSTOM };

enum class BufferAttrType {
  UINT = 0,

  INT,
  VEC2I,
  VEC3I,
  VEC4I,

  FLOAT,
  VEC2F,
  VEC3F,
  VEC4F,

  DOUBLE,
  VEC2D,
  VEC3D,
  VEC4D,

  UINT64,
  VEC2UI64,
  VEC3UI64,
  VEC4UI64,

  INT64,
  VEC2I64,
  VEC3I64,
  VEC4I64,

  // COLOR_R,
  // COLOR_RG,
  // COLOR_RGB,
  // COLOR_RGBA,

  MAX_BUFFER_ATTR_TYPE  // ALWAYS LEAVE THIS LAST, AND DO NOT USE
};

static std::array<TypeGLUqPtr, static_cast<size_t>(BufferAttrType::MAX_BUFFER_ATTR_TYPE)> attrTypeInfo = {{
    TypeGLUqPtr(new TypeGL<unsigned int, 1>()),  // UINT

    TypeGLUqPtr(new TypeGL<int, 1>()),  // INT
    TypeGLUqPtr(new TypeGL<int, 2>()),  // VEC2I
    TypeGLUqPtr(new TypeGL<int, 3>()),  // VEC3I
    TypeGLUqPtr(new TypeGL<int, 4>()),  // VEC4I

    TypeGLUqPtr(new TypeGL<float, 1>()),  // FLOAT
    TypeGLUqPtr(new TypeGL<float, 2>()),  // VEC2F
    TypeGLUqPtr(new TypeGL<float, 3>()),  // VEC3F
    TypeGLUqPtr(new TypeGL<float, 4>()),  // VEC4F

    TypeGLUqPtr(new TypeGL<double, 1>()),  // DOUBLE
    TypeGLUqPtr(new TypeGL<double, 2>()),  // VEC2D
    TypeGLUqPtr(new TypeGL<double, 3>()),  // VEC3D
    TypeGLUqPtr(new TypeGL<double, 4>()),  // VEC4D

    TypeGLUqPtr(new TypeGL<uint64_t, 1>()),  // UINT64
    TypeGLUqPtr(new TypeGL<uint64_t, 2>()),  // VEC2UI64
    TypeGLUqPtr(new TypeGL<uint64_t, 3>()),  // VEC3UI64
    TypeGLUqPtr(new TypeGL<uint64_t, 4>()),  // VEC4UI64

    TypeGLUqPtr(new TypeGL<int64_t, 1>()),  // INT64
    TypeGLUqPtr(new TypeGL<int64_t, 2>()),  // VEC2I64
    TypeGLUqPtr(new TypeGL<int64_t, 3>()),  // VEC3I64
    TypeGLUqPtr(new TypeGL<int64_t, 4>()),  // VEC4I64

    // TypeGLUqPtr(new TypeGL<uint8_t, 1>(true, true)),   // COLOR_R
    // TypeGLUqPtr(new TypeGL<uint8_t, 2>(true, true)),   // COLOR_RG
    // TypeGLUqPtr(new TypeGL<uint8_t, 3>(true, true)),   // COLOR_RGB
    // TypeGLUqPtr(new TypeGL<uint8_t, 4>(true, true))    // COLOR_RGBA
}

};

BufferAttrType getBufferAttrType(unsigned int a, int numComponents = 1);
BufferAttrType getBufferAttrType(int a, int numComponents = 1);
BufferAttrType getBufferAttrType(float a, int numComponents = 1);
BufferAttrType getBufferAttrType(double a, int numComponents = 1);

struct BufferAttrInfo {
  std::string name;
  BufferAttrType type;
  BaseTypeGL* typeInfo;
  int stride;
  int offset;

  BufferAttrInfo(const std::string& name, BufferAttrType type, BaseTypeGL* typeInfo, int stride, int offset)
      : name(name), type(type), typeInfo(typeInfo), stride(stride), offset(offset) {}
};

// tags for boost::multi_index_container
struct name {};

typedef std::unique_ptr<BufferAttrInfo> BufferAttrInfoPtr;

class BaseBufferLayout {
 public:
  BaseBufferLayout(LayoutType layoutType) : _layoutType(layoutType) {}
  virtual ~BaseBufferLayout() {}

  bool hasAttribute(const std::string& attrName) {
    BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
    return (nameLookup.find(attrName) != nameLookup.end());
    // return (_attrMap.find(attrName) != _attrMap.end());
  }

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) {
    BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
    BufferAttrMap_by_name::iterator itr;

    // TODO: throw an exception instead of an assert
    assert((itr = nameLookup.find(attrName)) != nameLookup.end());

    return (*itr)->typeInfo->clone();
  }

  BufferAttrType getAttributeType(const std::string& attrName) {
    // TODO(croot): consolidate this code and the one in getAttributeTypeGL()
    // into a single getBufferAttrInfo func or something.
    BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
    BufferAttrMap_by_name::iterator itr;

    // TODO(croot): throw an exception instead of an assert
    assert((itr = nameLookup.find(attrName)) != nameLookup.end());

    return (*itr)->type;
  }

  virtual void bindToRenderer(Shader* activeShader,
                              int numActiveBufferItems,
                              const std::string& attr = "",
                              const std::string& shaderAttr = "") = 0;

 protected:
  // typedef std::unordered_map<std::string, BufferAttrInfoPtr> BufferAttrMap;

  typedef multi_index_container<
      BufferAttrInfoPtr,
      indexed_by<random_access<>,

                 // hashed on name
                 hashed_unique<tag<name>, member<BufferAttrInfo, std::string, &BufferAttrInfo::name>>>> BufferAttrMap;

  typedef BufferAttrMap::index<name>::type BufferAttrMap_by_name;

  LayoutType _layoutType;
  BufferAttrMap _attrMap;
};

typedef std::shared_ptr<BaseBufferLayout> BufferLayoutShPtr;
typedef std::unique_ptr<BaseBufferLayout> BufferLayoutUqPtr;

class CustomBufferLayout : public BaseBufferLayout {
 public:
  CustomBufferLayout() : BaseBufferLayout(CUSTOM) {}

  void addAttribute(const std::string& attrName, BufferAttrType type, int stride, int offset) {
    // TODO: throw exception instead
    assert(!hasAttribute(attrName) && attrName.length());

    // _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(attrName, type,
    // attrTypeInfo[static_cast<int>(type)].get(), stride, offset));

    // _attrMap.emplace_back(new BufferAttrInfo(attrName, type, attrTypeInfo[static_cast<int>(type)].get(), stride,
    // offset));
    _attrMap.push_back(BufferAttrInfoPtr(
        new BufferAttrInfo(attrName, type, attrTypeInfo[static_cast<int>(type)].get(), stride, offset)));
  }

  void bindToRenderer(Shader* activeShader,
                      int numActiveBufferItems,
                      const std::string& attr = "",
                      const std::string& shaderAttr = "") {}
};

class InterleavedBufferLayout : public BaseBufferLayout {
 public:
  InterleavedBufferLayout() : BaseBufferLayout(INTERLEAVED), _vertexByteSize(0) {}

  int getBytesPerVertex() { return _vertexByteSize; }

  void addAttribute(const std::string& attrName, BufferAttrType type) {
    // TODO: throw exception instead
    assert(!hasAttribute(attrName) && attrName.length());

    // TODO, set the stride of all currently existing attrs, or leave
    // that for when the layout is bound to the renderer/shader/VAO

    int enumVal = static_cast<int>(type);
    // _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), -1,
    // _vertexByteSize));

    // _attrMap.emplace_back(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), -1, _vertexByteSize));
    _attrMap.push_back(
        BufferAttrInfoPtr(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), -1, _vertexByteSize)));

    _vertexByteSize += attrTypeInfo[enumVal]->numBytes();
  }

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName) {
    // TODO(croot): throw exception
    assert(!hasAttribute(attrName) && attrName.length());

    BufferAttrType type = getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);

    _attrMap.push_back(
        BufferAttrInfoPtr(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), -1, _vertexByteSize)));

    _vertexByteSize += attrTypeInfo[enumVal]->numBytes();
  }

  void bindToRenderer(Shader* activeShader,
                      int numActiveBufferItems,
                      const std::string& attr = "",
                      const std::string& shaderAttr = "") {
    GLuint attrLoc;
    BaseTypeGL* attrPtr;

    BufferAttrInfo* bufAttrPtr;

    if (!attr.length()) {
      BufferAttrMap::iterator itr;
      for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
        bufAttrPtr = itr->get();
        attrLoc = activeShader->getVertexAttributeLocation(bufAttrPtr->name);
        attrPtr = bufAttrPtr->typeInfo;
        attrPtr->bind(attrLoc, _vertexByteSize, bufAttrPtr->offset);
      }
    } else {
      // TODO(croot): throw an exception
      BufferAttrMap_by_name& nameLookup = _attrMap.get<name>();
      BufferAttrMap_by_name::iterator itr;

      // TODO(croot): throw an exception instead of an assert
      assert((itr = nameLookup.find(attr)) != nameLookup.end());
      bufAttrPtr = itr->get();
      attrLoc = activeShader->getVertexAttributeLocation(shaderAttr.length() ? shaderAttr : bufAttrPtr->name);
      attrPtr = bufAttrPtr->typeInfo;
      attrPtr->bind(attrLoc, _vertexByteSize, bufAttrPtr->offset);
    }
  }

 private:
  int _vertexByteSize;
};

class SequentialBufferLayout : public BaseBufferLayout {
 public:
  SequentialBufferLayout() : BaseBufferLayout(SEQUENTIAL) {}

  void addAttribute(const std::string& attrName, BufferAttrType type) {
    // TODO: throw exception instead
    assert(!hasAttribute(attrName) && attrName.length());

    int enumVal = static_cast<int>(type);
    // _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(),
    // attrTypeInfo[enumVal]->numBytes(), -1));
    // _attrMap.emplace_back(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(),
    // attrTypeInfo[enumVal]->numBytes(), -1));
    _attrMap.push_back(BufferAttrInfoPtr(
        new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), attrTypeInfo[enumVal]->numBytes(), -1)));
  }

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName) {
    // TODO: throw exception
    assert(!hasAttribute(attrName) && attrName.length());

    BufferAttrType type = getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);
    // _attrMap[attrName] = BufferAttrInfoPtr(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(),
    // attrTypeInfo[enumVal]->numBytes(), -1));
    // _attrMap.emplace_back(new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(),
    // attrTypeInfo[enumVal]->numBytes(), -1));
    _attrMap.push_back(BufferAttrInfoPtr(
        new BufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), attrTypeInfo[enumVal]->numBytes(), -1)));
  }

  void bindToRenderer(Shader* activeShader,
                      int numActiveBufferItems,
                      const std::string& attr = "",
                      const std::string& shaderAttr = "") {
    GLuint attrLoc;
    BaseTypeGL* attrPtr;
    int offset = 0;

    BufferAttrInfo* bufAttrPtr;

    BufferAttrMap::iterator itr;

    if (!attr.length()) {
      for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
        bufAttrPtr = itr->get();
        attrLoc = activeShader->getVertexAttributeLocation(bufAttrPtr->name);
        attrPtr = bufAttrPtr->typeInfo;
        // glVertexAttribPointer(attrLoc, attrPtr->numComponents(), attrPtr->baseGLType(), GL_FALSE, bufAttrPtr->stride,
        // BUFFER_OFFFSET(offset));
        attrPtr->bind(attrLoc, bufAttrPtr->stride, offset);
        glEnableVertexAttribArray(attrLoc);
        offset += attrPtr->numBytes() * numActiveBufferItems;
      }
    } else {
      // TODO: throw an exception
      assert(hasAttribute(attr));

      for (itr = _attrMap.begin(); itr != _attrMap.end(); ++itr) {
        bufAttrPtr = itr->get();
        attrPtr = bufAttrPtr->typeInfo;

        if (bufAttrPtr->name == attr) {
          attrLoc = activeShader->getVertexAttributeLocation(shaderAttr.length() ? shaderAttr : bufAttrPtr->name);
          // glVertexAttribPointer(attrLoc, attrPtr->numComponents(), attrPtr->baseGLType(), GL_FALSE,
          // bufAttrPtr->stride, BUFFER_OFFSET(offset));
          attrPtr->bind(attrLoc, bufAttrPtr->stride, offset);
          glEnableVertexAttribArray(attrLoc);
          break;
        }

        offset += attrPtr->numBytes() * numActiveBufferItems;
      }
    }

    // for (const auto& itr : _attrMap) {
    //     attrLoc = activeShader->getVertexAttributeLocation(itr.first);
    //     attrPtr = itr.second->typeInfo;
    //     glVertexAttribPointer(attrLoc, attrPtr->numComponents(), attrPtr->baseGLType(), GL_FALSE, itr.second->stride,
    //     BUFFER_OFFSET(offset));
    //     glEnableVertexAttribArray(attrLoc);
    //     offset += attrPtr->numBytes() * numActiveBufferItems;
    // }
  }
};

}  // namespace MapD_Renderer

#endif  // BUFFER_LAYOUT_H_
