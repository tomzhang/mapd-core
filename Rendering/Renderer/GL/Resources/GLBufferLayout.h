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

  MAX_GL_BUFFER_ATTR_TYPE  // ALWAYS LEAVE THIS LAST, AND DO NOT USE
};

namespace detail {
static std::array<TypeGLUqPtr, static_cast<size_t>(GLBufferAttrType::MAX_GL_BUFFER_ATTR_TYPE)> attrTypeInfo = {{
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
}};

GLBufferAttrType getBufferAttrType(unsigned int a, int numComponents = 1);
GLBufferAttrType getBufferAttrType(int a, int numComponents = 1);
GLBufferAttrType getBufferAttrType(float a, int numComponents = 1);
GLBufferAttrType getBufferAttrType(double a, int numComponents = 1);

}  // namespace detail

namespace MultiIndexTags {

// tags for boost::multi_index_container
struct name {};

}  // namespace MultiIndexTags

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
  virtual ~GLBaseBufferLayout() {}

  size_t getNumBytesPerVertex() const { return _vertexByteSize; }

  bool hasAttribute(const std::string& attrName) const;

  TypeGLShPtr getAttributeTypeGL(const std::string& attrName) const;

  GLBufferAttrType getAttributeType(const std::string& attrName) const;

  const GLBufferAttrInfo& getAttributeInfo(const std::string& attrName) const;
  int getAttributeByteOffset(const std::string& attrName) const;

  // TODO(croot): add an iterator to iterate over the attributes?

  inline int numAttributes() const { return _attrMap.size(); }

  const GLBufferAttrInfo& operator[](size_t i) const;

 protected:
  GLBaseBufferLayout(GLBufferLayoutType layoutType) : _layoutType(layoutType), _attrMap(), _vertexByteSize(0) {}

  typedef std::unique_ptr<GLBufferAttrInfo> BufferAttrInfoPtr;
  typedef boost::multi_index_container<
      BufferAttrInfoPtr,
      boost::multi_index::indexed_by<
          boost::multi_index::random_access<>,

          // hashed on name
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<MultiIndexTags::name>,
              boost::multi_index::member<GLBufferAttrInfo, std::string, &GLBufferAttrInfo::name>>>> BufferAttrMap;

  typedef BufferAttrMap::index<MultiIndexTags::name>::type BufferAttrMap_by_name;

  GLBufferLayoutType _layoutType;
  BufferAttrMap _attrMap;
  size_t _vertexByteSize;

 private:
  virtual void bindToShader(GLShader* activeShader,
                            int numActiveBufferItems,
                            const std::string& attr = "",
                            const std::string& shaderAttr = "") = 0;

  friend class ::Rendering::GL::Resources::GLVertexBuffer;
};

class GLCustomBufferLayout : public GLBaseBufferLayout {
 public:
  GLCustomBufferLayout() : GLBaseBufferLayout(GLBufferLayoutType::CUSTOM) {}
  ~GLCustomBufferLayout() {}

  void addAttribute(const std::string& attrName, GLBufferAttrType type, int stride, int offset);

 private:
  void bindToShader(GLShader* activeShader,
                    int numActiveBufferItems,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "") final {
    // TODO(croot): implement
    THROW_RUNTIME_EX("CustomBufferLayout::bindToShader() has yet to be implemented.");
  }
};

class GLInterleavedBufferLayout : public GLBaseBufferLayout {
 public:
  GLInterleavedBufferLayout() : GLBaseBufferLayout(GLBufferLayoutType::INTERLEAVED) {}

  void addAttribute(const std::string& attrName, GLBufferAttrType type);

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName) {
    RUNTIME_EX_ASSERT(
        !hasAttribute(attrName) && attrName.length(),
        "GLInterleavedBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

    GLBufferAttrType type = detail::getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);

    _attrMap.push_back(BufferAttrInfoPtr(
        new GLBufferAttrInfo(attrName, type, detail::attrTypeInfo[enumVal].get(), -1, _vertexByteSize)));

    _vertexByteSize += detail::attrTypeInfo[enumVal]->numBytes();
  }

 private:
  void bindToShader(GLShader* activeShader,
                    int numActiveBufferItems,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "") final;
};

class GLSequentialBufferLayout : public GLBaseBufferLayout {
 public:
  GLSequentialBufferLayout() : GLBaseBufferLayout(GLBufferLayoutType::SEQUENTIAL) {}

  void addAttribute(const std::string& attrName, GLBufferAttrType type);

  template <typename T, int numComponents = 1>
  void addAttribute(const std::string& attrName) {
    RUNTIME_EX_ASSERT(
        !hasAttribute(attrName) && attrName.length(),
        "GLSequentialBufferLayout::addAttribute(): attribute " + attrName + " already exists in the layout.");

    GLBufferAttrType type = detail::getBufferAttrType(T(0), numComponents);
    int enumVal = static_cast<int>(type);
    // _attrMap[attrName] = BufferAttrInfoPtr(new GLBufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(),
    // attrTypeInfo[enumVal]->numBytes(), -1));
    // _attrMap.emplace_back(new GLBufferAttrInfo(attrName, type, attrTypeInfo[enumVal].get(),
    // attrTypeInfo[enumVal]->numBytes(), -1));
    _attrMap.push_back(BufferAttrInfoPtr(new GLBufferAttrInfo(
        attrName, type, detail::attrTypeInfo[enumVal].get(), detail::attrTypeInfo[enumVal]->numBytes(), -1)));

    _vertexByteSize += detail::attrTypeInfo[enumVal]->numBytes();
  }

 private:
  void bindToShader(GLShader* activeShader,
                    int numActiveBufferItems,
                    const std::string& attr = "",
                    const std::string& shaderAttr = "") final;
};

}  // namespace Resources
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_RESOURCES_GLBUFFERLAYOUT_H_
