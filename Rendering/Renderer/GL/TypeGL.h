#ifndef RENDERING_GL_TYPEGL_H_
#define RENDERING_GL_TYPEGL_H_

#include "Types.h"
#include "MapDGL.h"
#include <GL/glew.h>
#include <string>
#include <memory>
#include <functional>
#include <cstdint>

namespace Rendering {
namespace GL {

struct BaseTypeGL {
  static std::string getTypeDefinesMacroForShader();

  BaseTypeGL() {}
  BaseTypeGL(const BaseTypeGL& baseTypeGL) {}
  virtual ~BaseTypeGL() {}

  virtual std::shared_ptr<BaseTypeGL> clone() const = 0;

  virtual int numComponents() const = 0;
  virtual int numBytes() const = 0;
  // virtual int numGLSLBytes() const = 0;
  virtual int baseGLType() const = 0;
  virtual int glslGLType() const = 0;
  virtual std::string glslType() const = 0;

  virtual void bind(GLuint shaderAttrLoc, int stride, int offset) const = 0;

  virtual GLboolean normalize() { return GL_FALSE; }

  inline friend bool operator==(const BaseTypeGL& lhs, const BaseTypeGL& rhs) {
    return (lhs.numComponents() == rhs.numComponents() && lhs.numBytes() == rhs.numBytes() &&
            lhs.baseGLType() == rhs.baseGLType());
  }

  inline friend bool operator!=(const BaseTypeGL& lhs, const BaseTypeGL& rhs) {
    return (lhs.numComponents() != rhs.numComponents() || lhs.numBytes() != rhs.numBytes() ||
            lhs.baseGLType() != rhs.baseGLType());
  }
};

template <typename T, int componentCnt = 1>
struct TypeGL : BaseTypeGL {
  TypeGL(bool useAsFloat = false, bool normalize = false) { setUseAsFloat(useAsFloat, normalize); }

  TypeGL(const TypeGL<T, componentCnt>& typeGL) { setUseAsFloat(typeGL._useAsFloat, typeGL._normalize); }

  ~TypeGL() {}

  TypeGLShPtr clone() const { return TypeGLShPtr(new TypeGL<T, componentCnt>(*this)); }

  int numComponents() const { return componentCnt; }

  int numBytes() const { return sizeof(T) * numComponents(); }

  // int numGLSLBytes() { return numBytes(); }

  int baseGLType() const final;
  int glslGLType() const final;
  std::string glslType() const;

  void setUseAsFloat(bool useAsFloat, bool normalize = false) {
    _useAsFloat = useAsFloat;
    _normalize = normalize;
    // if (useAsFloat) {
    //     _bindFunc = std::bind(glVertexAttribPointer, std::placeholders::_1, numComponents(), baseGLType(), (normalize
    //     ? GL_TRUE : GL_FALSE), std::placeholders::_2, std::placeholders::_3);

    //     _useAsFloat = useAsFloat;
    //     _normalize = normalize;
    // } else {
    //     _bindFunc = std::bind(_getAltVertexAttribPointerFunc(), std::placeholders::_1, numComponents(), baseGLType(),
    //     std::placeholders::_2, std::placeholders::_3);
    //     _normalize = false;
    // }
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

  // void bind(GLuint shaderAttrLoc, int stride, int offset) {
  //     _bindFunc(shaderAttrLoc, stride, BUFFER_OFFSET(offset));
  // }

 private:
  typedef std::function<void(GLuint, GLint, GLenum, GLsizei, const GLvoid*)> VertexAttribPtrFunc;

  bool _useAsFloat;  // when binding the attr type, this determines whether the attr
                     // will be bound as a float (via glVertexAttribPointer) or
                     // as another type (either glVertexAttribIPointer or glVertexAttribLPointer
                     // depending on the type).

  bool _normalize;  // if binding as a float, this determines whether the value when converted
                    // to a float should be normalized. If true, if the values is unsigned
                    // normalization results in a value from 0 to 1. If signed, then it
                    // results in a value from -1 to 1.

  // std::function<void(GLuint, GLsizei, const GLvoid *)> _bindFunc;

  VertexAttribPtrFunc _getAltVertexAttribPointerFunc() const { return glVertexAttribIPointer; }
};

// SPECIALIZATIONS

// BOOL:

template <>
std::string TypeGL<bool, 1>::glslType() const;

template <>
int TypeGL<bool, 1>::baseGLType() const;

template <>
int TypeGL<bool, 1>::glslGLType() const;

template <>
std::string TypeGL<bool, 2>::glslType() const;

template <>
int TypeGL<bool, 2>::baseGLType() const;

template <>
int TypeGL<bool, 2>::glslGLType() const;

template <>
std::string TypeGL<bool, 3>::glslType() const;

template <>
int TypeGL<bool, 3>::baseGLType() const;

template <>
int TypeGL<bool, 3>::glslGLType() const;

template <>
std::string TypeGL<bool, 4>::glslType() const;

template <>
int TypeGL<bool, 4>::baseGLType() const;

template <>
int TypeGL<bool, 4>::glslGLType() const;

// UNSIGNED INT:

template <>
std::string TypeGL<unsigned int, 1>::glslType() const;

template <>
int TypeGL<unsigned int, 1>::baseGLType() const;

template <>
int TypeGL<unsigned int, 1>::glslGLType() const;

// INT:

template <>
std::string TypeGL<int, 1>::glslType() const;

template <>
int TypeGL<int, 1>::baseGLType() const;

template <>
int TypeGL<int, 1>::glslGLType() const;

template <>
std::string TypeGL<int, 2>::glslType() const;

template <>
int TypeGL<int, 2>::baseGLType() const;

template <>
int TypeGL<int, 2>::glslGLType() const;

template <>
std::string TypeGL<int, 3>::glslType() const;

template <>
int TypeGL<int, 3>::baseGLType() const;

template <>
int TypeGL<int, 3>::glslGLType() const;

template <>
std::string TypeGL<int, 4>::glslType() const;

template <>
int TypeGL<int, 4>::baseGLType() const;

template <>
int TypeGL<int, 4>::glslGLType() const;

// FLOAT

template <>
std::string TypeGL<float, 1>::glslType() const;

template <>
int TypeGL<float, 1>::baseGLType() const;

template <>
int TypeGL<float, 1>::glslGLType() const;

template <>
void TypeGL<float, 1>::setUseAsFloat(bool bindAsFloat, bool normalize);

template <>
std::string TypeGL<float, 2>::glslType() const;

template <>
int TypeGL<float, 2>::baseGLType() const;

template <>
int TypeGL<float, 2>::glslGLType() const;

template <>
void TypeGL<float, 2>::setUseAsFloat(bool bindAsFloat, bool normalize);

template <>
std::string TypeGL<float, 3>::glslType() const;

template <>
int TypeGL<float, 3>::baseGLType() const;

template <>
int TypeGL<float, 3>::glslGLType() const;

template <>
void TypeGL<float, 3>::setUseAsFloat(bool bindAsFloat, bool normalize);

template <>
std::string TypeGL<float, 4>::glslType() const;

template <>
int TypeGL<float, 4>::baseGLType() const;

template <>
int TypeGL<float, 4>::glslGLType() const;

template <>
void TypeGL<float, 4>::setUseAsFloat(bool bindAsFloat, bool normalize);

// DOUBLE

template <>
std::string TypeGL<double, 1>::glslType() const;

template <>
int TypeGL<double, 1>::baseGLType() const;

template <>
int TypeGL<double, 1>::glslGLType() const;

template <>
TypeGL<double, 1>::VertexAttribPtrFunc TypeGL<double, 1>::_getAltVertexAttribPointerFunc() const;

template <>
std::string TypeGL<double, 2>::glslType() const;

template <>
int TypeGL<double, 2>::baseGLType() const;

template <>
int TypeGL<double, 2>::glslGLType() const;

template <>
TypeGL<double, 2>::VertexAttribPtrFunc TypeGL<double, 2>::_getAltVertexAttribPointerFunc() const;

template <>
std::string TypeGL<double, 3>::glslType() const;

template <>
int TypeGL<double, 3>::baseGLType() const;

template <>
int TypeGL<double, 3>::glslGLType() const;

template <>
TypeGL<double, 3>::VertexAttribPtrFunc TypeGL<double, 3>::_getAltVertexAttribPointerFunc() const;

template <>
std::string TypeGL<double, 4>::glslType() const;

template <>
int TypeGL<double, 4>::baseGLType() const;

template <>
int TypeGL<double, 4>::glslGLType() const;

template <>
TypeGL<double, 4>::VertexAttribPtrFunc TypeGL<double, 4>::_getAltVertexAttribPointerFunc() const;

// UINT8

template <>
std::string TypeGL<uint8_t, 1>::glslType() const;

template <>
int TypeGL<uint8_t, 1>::baseGLType() const;

template <>
int TypeGL<uint8_t, 1>::glslGLType() const;

template <>
std::string TypeGL<uint8_t, 2>::glslType() const;

template <>
int TypeGL<uint8_t, 2>::baseGLType() const;

template <>
int TypeGL<uint8_t, 2>::glslGLType() const;

template <>
std::string TypeGL<uint8_t, 3>::glslType() const;

template <>
int TypeGL<uint8_t, 3>::baseGLType() const;

template <>
int TypeGL<uint8_t, 3>::glslGLType() const;
template <>
std::string TypeGL<uint8_t, 4>::glslType() const;

template <>
int TypeGL<uint8_t, 4>::baseGLType() const;

template <>
int TypeGL<uint8_t, 4>::glslGLType() const;

// UINT64

template <>
std::string TypeGL<uint64_t, 1>::glslType() const;

template <>
int TypeGL<uint64_t, 1>::baseGLType() const;

template <>
int TypeGL<uint64_t, 1>::glslGLType() const;

// template <>
// int TypeGL<uint64_t, 1>::numGLSLBytes();

template <>
std::string TypeGL<uint64_t, 2>::glslType() const;

template <>
int TypeGL<uint64_t, 2>::baseGLType() const;

template <>
int TypeGL<uint64_t, 2>::glslGLType() const;

// template <>
// int TypeGL<uint64_t, 2>::numGLSLBytes();

template <>
std::string TypeGL<uint64_t, 3>::glslType() const;

template <>
int TypeGL<uint64_t, 3>::baseGLType() const;

template <>
int TypeGL<uint64_t, 3>::glslGLType() const;

// template <>
// int TypeGL<uint64_t, 3>::numGLSLBytes();

template <>
std::string TypeGL<uint64_t, 4>::glslType() const;

template <>
int TypeGL<uint64_t, 4>::baseGLType() const;

template <>
int TypeGL<uint64_t, 4>::glslGLType() const;

// template <>
// int TypeGL<uint64_t, 4>::numGLSLBytes();

// INT64

template <>
std::string TypeGL<int64_t, 1>::glslType() const;

template <>
int TypeGL<int64_t, 1>::baseGLType() const;

template <>
int TypeGL<int64_t, 1>::glslGLType() const;

// template <>
// int TypeGL<int64_t, 1>::numGLSLBytes();

template <>
std::string TypeGL<int64_t, 2>::glslType() const;

template <>
int TypeGL<int64_t, 2>::baseGLType() const;

template <>
int TypeGL<int64_t, 2>::glslGLType() const;

// template <>
// int TypeGL<int64_t, 2>::numGLSLBytes();

template <>
std::string TypeGL<int64_t, 3>::glslType() const;

template <>
int TypeGL<int64_t, 3>::baseGLType() const;

template <>
int TypeGL<int64_t, 3>::glslGLType() const;

// template <>
// int TypeGL<int64_t, 3>::numGLSLBytes();

template <>
std::string TypeGL<int64_t, 4>::glslType() const;

template <>
int TypeGL<int64_t, 4>::baseGLType() const;

template <>
int TypeGL<int64_t, 4>::glslGLType() const;

// template <>
// int TypeGL<int64_t, 4>::numGLSLBytes();

}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_TYPEGL_H_
