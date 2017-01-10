#ifndef RENDERING_GL_RESOURCES_GLBUFFERLAYOUT_H_
#define RENDERING_GL_RESOURCES_GLBUFFERLAYOUT_H_

#include "Types.h"
#include "../Types.h"
#include "../../../RenderError.h"
#include "../TypeGL.h"
#include <array>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>

namespace Rendering {
namespace GL {
namespace Resources {

enum class GLBufferLayoutType { INTERLEAVED = 0, SEQUENTIAL, CUSTOM };

enum class GLBufferAttrType {
  UINT = 0,
  VEC2UI,
  VEC3UI,
  VEC4UI,

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

  MAX_GL_BUFFER_ATTR_TYPE  // ALWAYS LEAVE THIS LAST, AND DO NOT USE
};

struct GLBufferAttrInfo {
  std::string name;
  GLBufferAttrType type;
  BaseTypeGL* typeInfo;
  int stride;
  int offset;

  GLBufferAttrInfo(const std::string& name, GLBufferAttrType type, BaseTypeGL* typeInfo, int stride, int offset)
      : name(name), type(type), typeInfo(typeInfo), stride(stride), offset(offset) {}

  bool operator==(const GLBufferAttrInfo& attrInfo) const {
    return (type == attrInfo.type && stride == attrInfo.stride && offset == attrInfo.offset);
  }

  bool operator!=(const GLBufferAttrInfo& attrInfo) const { return !operator==(attrInfo); }
};

class GLBaseBufferLayout {
 public:
  GLBaseBufferLayout(const GLBaseBufferLayout& layout);
  virtual ~GLBaseBufferLayout() {}

  GLBufferLayoutType getLayoutType() const { return _layoutType; }
  virtual size_t getNumBytesPerItem() const { return _itemByteSize; }

  bool hasAttribute(const std::string& attrName) const;

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const;

  GLBufferAttrType getAttributeType(const std::string& attrName) const;

  const GLBufferAttrInfo& getAttributeInfo(const std::string& attrName) const;
  int getAttributeByteOffset(const std::string& attrName) const;

  // TODO(croot): add an iterator to iterate over the attributes?
  inline int numAttributes() const { return _attrMap.size(); }
  const GLBufferAttrInfo& operator[](size_t i) const;

  virtual bool operator==(const GLBaseBufferLayout& layoutPtr) const;

  virtual bool operator!=(const GLBaseBufferLayout& layoutPtr) const { return !operator==(layoutPtr); }

 protected:
  GLBaseBufferLayout(const std::set<std::string>& supportedExtensions, GLBufferLayoutType layoutType);

  // tags for boost::multi_index_container
  struct name {};
  struct rsrcid {};

  typedef std::unique_ptr<GLBufferAttrInfo> BufferAttrInfoPtr;
  typedef boost::multi_index_container<
      BufferAttrInfoPtr,
      boost::multi_index::indexed_by<
          boost::multi_index::random_access<>,

          // hashed on name
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<name>,
              boost::multi_index::member<GLBufferAttrInfo, std::string, &GLBufferAttrInfo::name>>>>
      BufferAttrMap;

  typedef BufferAttrMap::index<name>::type BufferAttrMap_by_name;

  GLBufferLayoutType _layoutType;
  BufferAttrMap _attrMap;

  size_t _itemByteSize;

  static std::array<TypeGLUqPtr, static_cast<size_t>(GLBufferAttrType::MAX_GL_BUFFER_ATTR_TYPE)> attrTypeInfo;
  static GLBufferAttrType getBufferAttrType(unsigned int a, int numComponents = 1);
  static GLBufferAttrType getBufferAttrType(int a, int numComponents = 1);
  static GLBufferAttrType getBufferAttrType(float a, int numComponents = 1);
  static GLBufferAttrType getBufferAttrType(double a, int numComponents = 1);
  static GLBufferAttrType getBufferAttrType(uint64_t a, int numComponents = 1);
  static GLBufferAttrType getBufferAttrType(int64_t a, int numComponents = 1);

 private:
  virtual void bindToShader(GLShader* activeShader,
                            const size_t usedBytes,
                            const size_t offsetBytes,
                            const std::string& attr = "",
                            const std::string& shaderAttr = "") const = 0;

  friend class ::Rendering::GL::Resources::GLLayoutManagerBuffer;
  friend class ::Rendering::GL::Resources::GLVertexBuffer;
};

class GLCustomBufferLayout : public GLBaseBufferLayout {
 public:
  GLCustomBufferLayout(const std::set<std::string>& supportedExtensions)
      : GLBaseBufferLayout(supportedExtensions, GLBufferLayoutType::CUSTOM) {}
  ~GLCustomBufferLayout() {}

  void addAttribute(const std::string& attrName, GLBufferAttrType type, int stride, int offset);

 private:
  void bindToShader(GLShader* activeShader,
                    const size_t usedBytes,
                    const size_t offsetBytes,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "") const final {
    // TODO(croot): implement
    THROW_RUNTIME_EX("CustomBufferLayout::bindToShader() has yet to be implemented.");
  }
};

class GLInterleavedBufferLayout : public GLBaseBufferLayout {
 public:
  GLInterleavedBufferLayout(const std::set<std::string>& supportedExtensions)
      : GLBaseBufferLayout(supportedExtensions, GLBufferLayoutType::INTERLEAVED) {}

  void addAttribute(const std::string& attrName, GLBufferAttrType type);

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName) {
    RUNTIME_EX_ASSERT(
        !hasAttribute(attrName) && attrName.length(),
        "GLInterleavedBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

    GLBufferAttrType type = getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);

    _attrMap.push_back(
        BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), -1, _itemByteSize)));

    _itemByteSize += attrTypeInfo[enumVal]->numBytes();
  }

 private:
  void bindToShader(GLShader* activeShader,
                    const size_t usedBytes,
                    const size_t offsetBytes,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "") const final;
};

class GLSequentialBufferLayout : public GLBaseBufferLayout {
 public:
  GLSequentialBufferLayout(const std::set<std::string>& supportedExtensions)
      : GLBaseBufferLayout(supportedExtensions, GLBufferLayoutType::SEQUENTIAL) {}

  void addAttribute(const std::string& attrName, GLBufferAttrType type);

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName) {
    RUNTIME_EX_ASSERT(
        !hasAttribute(attrName) && attrName.length(),
        "GLSequentialBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

    GLBufferAttrType type = getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);
    _attrMap.push_back(BufferAttrInfoPtr(
        new GLBufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(), attrTypeInfo[enumVal]->numBytes(), -1)));

    _itemByteSize += attrTypeInfo[enumVal]->numBytes();
  }

 private:
  void bindToShader(GLShader* activeShader,
                    const size_t usedBytes,
                    const size_t offsetBytes,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "") const final;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLBUFFERLAYOUT_H_
