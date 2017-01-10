#ifndef RENDERING_GL_TYPEGL_H_
#define RENDERING_GL_TYPEGL_H_

#include "Types.h"
#include "MapDGL.h"
#include <GL/glew.h>
#include <string>
#include <memory>
#include <functional>
#include <cstdint>
#include <sstream>
#include <set>

namespace Rendering {
namespace GL {

std::string gl_type_to_string(const int glType);

struct BaseTypeGL {
  static std::string getTypeDefinesMacroForShader();
  static std::string getExtensionStr();

  template <class T>
  struct is_convertible_to_float
      : std::integral_constant<bool,
                               std::is_same<int, T>::value || std::is_same<unsigned int, T>::value ||
                                   std::is_same<double, T>::value> {};

  virtual ~BaseTypeGL() {}

  virtual TypeGLShPtr clone() const = 0;

  virtual int numComponents() const = 0;
  virtual int numBytes() const = 0;
  int baseGLType() const { return _baseGLType; }
  int glslGLType() const { return _glslGLType; }
  std::string glslType() const {
    if (_useAsFloat) {
      auto numComps = numComponents();
      if (numComps == 1) {
        return "float";
      } else {
        return "vec" + std::to_string(numComps);
      }
    }
    return _glslType;
  }

  void bind(GLuint shaderAttrLoc, int stride, int offset) const {
    if (_useAsFloat) {
      MAPD_CHECK_GL_ERROR(glVertexAttribPointer(shaderAttrLoc,
                                                numComponents(),
                                                baseGLType(),
                                                (_normalize ? GL_TRUE : GL_FALSE),
                                                stride,
                                                BUFFER_OFFSET(offset)));
    } else {
      MAPD_CHECK_GL_ERROR(_getAltVertexAttribPointerFunc()(
          shaderAttrLoc, numComponents(), baseGLType(), stride, BUFFER_OFFSET(offset)));
    }
    MAPD_CHECK_GL_ERROR(glEnableVertexAttribArray(shaderAttrLoc));
  }

  inline friend bool operator==(const BaseTypeGL& lhs, const BaseTypeGL& rhs) {
    return (lhs.numComponents() == rhs.numComponents() && lhs.numBytes() == rhs.numBytes() &&
            lhs.baseGLType() == rhs.baseGLType());
  }

  inline friend bool operator!=(const BaseTypeGL& lhs, const BaseTypeGL& rhs) {
    return (lhs.numComponents() != rhs.numComponents() || lhs.numBytes() != rhs.numBytes() ||
            lhs.baseGLType() != rhs.baseGLType());
  }

  operator std::string() const {
    std::ostringstream s;
    s << "<GL type: " << gl_type_to_string(baseGLType()) << ", Num components: " << numComponents()
      << ", Num bytes: " << numBytes() << ">";
    return s.str();
  }

 protected:
  typedef std::function<void(GLuint, GLint, GLenum, GLsizei, const GLvoid*)> VertexAttribPtrFunc;

  int _baseGLType;
  int _glslGLType;
  std::string _glslType;

  bool _useAsFloat;  // when binding the attr type, this determines whether the attr
                     // will be bound as a float (via glVertexAttribPointer) or
                     // as another type (either glVertexAttribIPointer or glVertexAttribLPointer
                     // depending on the type).

  bool _normalize;  // if binding as a float, this determines whether the value when converted
                    // to a float should be normalized. If true, if the values is unsigned
                    // normalization results in a value from 0 to 1. If signed, then it
                    // results in a value from -1 to 1.

  BaseTypeGL(const int baseGLType,
             const int glslGLType,
             const std::string& glslType,
             const bool useAsFloat = false,
             const bool normalize = false)
      : _baseGLType(baseGLType),
        _glslGLType(glslGLType),
        _glslType(glslType),
        _useAsFloat(useAsFloat),
        _normalize(normalize) {}

  BaseTypeGL(const BaseTypeGL& baseTypeGL)
      : _baseGLType(baseTypeGL._baseGLType),
        _glslGLType(baseTypeGL._glslGLType),
        _glslType(baseTypeGL._glslType),
        _useAsFloat(baseTypeGL._useAsFloat),
        _normalize(baseTypeGL._normalize) {}

  void _setUseAsFloat(const bool useAsFloat, const bool normalize) {
    _useAsFloat = useAsFloat;
    _normalize = normalize;
  }

 private:
  virtual VertexAttribPtrFunc _getAltVertexAttribPointerFunc() const = 0;
};

template <typename T>
struct TypeGLMiddleLayer : BaseTypeGL {
  virtual ~TypeGLMiddleLayer() {}

 protected:
  TypeGLMiddleLayer(const int baseGLType,
                    const int glslGLType,
                    const std::string& glslType,
                    const bool useAsFloat = false,
                    const bool normalize = false)
      : BaseTypeGL(baseGLType, glslGLType, glslType, useAsFloat, normalize) {}

  TypeGLMiddleLayer(const TypeGLMiddleLayer<T>& typeGLMiddleLayer) : BaseTypeGL(typeGLMiddleLayer) {}

 private:
  VertexAttribPtrFunc _getAltVertexAttribPointerFunc() const final { return glVertexAttribIPointer; }
};

template <>
TypeGLMiddleLayer<float>::TypeGLMiddleLayer(const int baseGLType,
                                            const int glslGLType,
                                            const std::string& glslType,
                                            const bool useAsFloat,
                                            const bool normalize);

template <>
TypeGLMiddleLayer<double>::VertexAttribPtrFunc TypeGLMiddleLayer<double>::_getAltVertexAttribPointerFunc() const;

template <>
TypeGLMiddleLayer<uint64_t>::VertexAttribPtrFunc TypeGLMiddleLayer<uint64_t>::_getAltVertexAttribPointerFunc() const;

template <>
TypeGLMiddleLayer<int64_t>::VertexAttribPtrFunc TypeGLMiddleLayer<int64_t>::_getAltVertexAttribPointerFunc() const;

template <typename T, int componentCnt, class Enable = void>
struct TypeGL : TypeGLMiddleLayer<T> {
  TypeGL(const std::set<std::string>& supportedExtensions);
  TypeGL(const TypeGL<T, componentCnt>& typeGL) : TypeGLMiddleLayer<T>(typeGL) {}
  ~TypeGL() {}

  TypeGLShPtr clone() const { return TypeGLShPtr(new TypeGL<T, componentCnt>(*this)); }

  int numComponents() const { return componentCnt; }
  int numBytes() const { return sizeof(T) * numComponents(); }
};

template <typename T, int componentCnt>
struct TypeGL<T, componentCnt, typename std::enable_if<BaseTypeGL::is_convertible_to_float<T>::value>::type>
    : TypeGLMiddleLayer<T> {
  TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat = false, bool normalize = false);
  TypeGL(const TypeGL<T, componentCnt>& typeGL) : TypeGLMiddleLayer<T>(typeGL) {}
  ~TypeGL() {}

  TypeGLShPtr clone() const { return TypeGLShPtr(new TypeGL<T, componentCnt>(*this)); }

  int numComponents() const { return componentCnt; }
  int numBytes() const { return sizeof(T) * numComponents(); }

  void setUseAsFloat(const bool useAsFloat, const bool normalize = false) {
    BaseTypeGL::_setUseAsFloat(useAsFloat, normalize);
  }
};

// SPECIALIZATIONS

// BOOL:

template <>
TypeGL<bool, 1>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<bool, 2>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<bool, 3>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<bool, 4>::TypeGL(const std::set<std::string>& supportedExtensions);

// UNSIGNED INT:

template <>
TypeGL<unsigned int, 1>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<unsigned int, 2>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<unsigned int, 3>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<unsigned int, 4>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

// INT:

template <>
TypeGL<int, 1>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<int, 2>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<int, 3>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<int, 4>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

// FLOAT

template <>
TypeGL<float, 1>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<float, 2>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<float, 3>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<float, 4>::TypeGL(const std::set<std::string>& supportedExtensions);

// DOUBLE

template <>
TypeGL<double, 1>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<double, 2>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<double, 3>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

template <>
TypeGL<double, 4>::TypeGL(const std::set<std::string>& supportedExtensions, bool useAsFloat, bool normalize);

// UINT64

template <>
TypeGL<uint64_t, 1>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<uint64_t, 2>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<uint64_t, 3>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<uint64_t, 4>::TypeGL(const std::set<std::string>& supportedExtensions);

// INT64

template <>
TypeGL<int64_t, 1>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<int64_t, 2>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<int64_t, 3>::TypeGL(const std::set<std::string>& supportedExtensions);

template <>
TypeGL<int64_t, 4>::TypeGL(const std::set<std::string>& supportedExtensions);

}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_TYPEGL_H_
