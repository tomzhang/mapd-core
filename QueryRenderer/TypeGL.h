#ifndef TYPE_GL_H_
#define TYPE_GL_H_

#include "MapDGL.h"

#include <GL/glew.h>
#include <string>
#include <memory>
#include <functional>
#include <cstdint>

namespace MapD_Renderer {

struct BaseTypeGL {
  BaseTypeGL() {}
  BaseTypeGL(const BaseTypeGL& baseTypeGL) {}
  virtual ~BaseTypeGL() {}

  virtual std::shared_ptr<BaseTypeGL> clone() = 0;

  virtual int numComponents() = 0;
  virtual int numBytes() = 0;
  // virtual int numGLSLBytes() = 0;
  virtual int baseGLType() = 0;
  virtual std::string glslType() = 0;

  virtual void bind(GLuint shaderAttrLoc, int stride, int offset) = 0;

  virtual GLboolean normalize() { return GL_FALSE; }
};

typedef std::unique_ptr<BaseTypeGL> TypeGLUqPtr;
typedef std::shared_ptr<BaseTypeGL> TypeGLShPtr;

template <typename T, int componentCnt = 1>
struct TypeGL : BaseTypeGL {
  TypeGL(bool useAsFloat = false, bool normalize = false) { setUseAsFloat(useAsFloat, normalize); }

  TypeGL(const TypeGL<T, componentCnt>& typeGL) { setUseAsFloat(typeGL._useAsFloat, typeGL._normalize); }

  ~TypeGL() {}

  TypeGLShPtr clone() { return TypeGLShPtr(new TypeGL<T, componentCnt>(*this)); }

  int numComponents() { return componentCnt; }

  int numBytes() { return sizeof(T) * numComponents(); }

  // int numGLSLBytes() { return numBytes(); }

  int baseGLType();
  std::string glslType();

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

  void bind(GLuint shaderAttrLoc, int stride, int offset) {
    if (_useAsFloat) {
      glVertexAttribPointer(shaderAttrLoc,
                            numComponents(),
                            baseGLType(),
                            (_normalize ? GL_TRUE : GL_FALSE),
                            stride,
                            BUFFER_OFFSET(offset));
    } else {
      _getAltVertexAttribPointerFunc()(shaderAttrLoc, numComponents(), baseGLType(), stride, BUFFER_OFFSET(offset));
    }
    glEnableVertexAttribArray(shaderAttrLoc);
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

  VertexAttribPtrFunc _getAltVertexAttribPointerFunc() { return glVertexAttribIPointer; }
};

// SPECIALIZATIONS

// UNSIGNED INT:

template <>
std::string TypeGL<unsigned int, 1>::glslType();

template <>
int TypeGL<unsigned int, 1>::baseGLType();

// INT:

template <>
std::string TypeGL<int, 1>::glslType();

template <>
int TypeGL<int, 1>::baseGLType();

template <>
std::string TypeGL<int, 2>::glslType();

template <>
int TypeGL<int, 2>::baseGLType();

template <>
std::string TypeGL<int, 3>::glslType();

template <>
int TypeGL<int, 3>::baseGLType();

template <>
std::string TypeGL<int, 4>::glslType();

template <>
int TypeGL<int, 4>::baseGLType();

// FLOAT

template <>
std::string TypeGL<float, 1>::glslType();

template <>
int TypeGL<float, 1>::baseGLType();

template <>
void TypeGL<float, 1>::setUseAsFloat(bool bindAsFloat, bool normalize);

template <>
std::string TypeGL<float, 2>::glslType();

template <>
int TypeGL<float, 2>::baseGLType();

template <>
void TypeGL<float, 2>::setUseAsFloat(bool bindAsFloat, bool normalize);

template <>
std::string TypeGL<float, 3>::glslType();

template <>
int TypeGL<float, 3>::baseGLType();

template <>
void TypeGL<float, 3>::setUseAsFloat(bool bindAsFloat, bool normalize);

template <>
std::string TypeGL<float, 4>::glslType();

template <>
int TypeGL<float, 4>::baseGLType();

template <>
void TypeGL<float, 4>::setUseAsFloat(bool bindAsFloat, bool normalize);

// DOUBLE

template <>
std::string TypeGL<double, 1>::glslType();

template <>
int TypeGL<double, 1>::baseGLType();

template <>
TypeGL<double, 1>::VertexAttribPtrFunc TypeGL<double, 1>::_getAltVertexAttribPointerFunc();

template <>
std::string TypeGL<double, 2>::glslType();

template <>
int TypeGL<double, 2>::baseGLType();

template <>
TypeGL<double, 2>::VertexAttribPtrFunc TypeGL<double, 2>::_getAltVertexAttribPointerFunc();

template <>
std::string TypeGL<double, 3>::glslType();

template <>
int TypeGL<double, 3>::baseGLType();

template <>
TypeGL<double, 3>::VertexAttribPtrFunc TypeGL<double, 3>::_getAltVertexAttribPointerFunc();

template <>
std::string TypeGL<double, 4>::glslType();

template <>
int TypeGL<double, 4>::baseGLType();

template <>
TypeGL<double, 4>::VertexAttribPtrFunc TypeGL<double, 4>::_getAltVertexAttribPointerFunc();

// UINT8

template <>
std::string TypeGL<uint8_t, 1>::glslType();

template <>
int TypeGL<uint8_t, 1>::baseGLType();

template <>
std::string TypeGL<uint8_t, 2>::glslType();

template <>
int TypeGL<uint8_t, 2>::baseGLType();

template <>
std::string TypeGL<uint8_t, 3>::glslType();

template <>
int TypeGL<uint8_t, 3>::baseGLType();
template <>
std::string TypeGL<uint8_t, 4>::glslType();

template <>
int TypeGL<uint8_t, 4>::baseGLType();

// UINT64

template <>
std::string TypeGL<uint64_t, 1>::glslType();

template <>
int TypeGL<uint64_t, 1>::baseGLType();

// template <>
// int TypeGL<uint64_t, 1>::numGLSLBytes();

template <>
std::string TypeGL<uint64_t, 2>::glslType();

template <>
int TypeGL<uint64_t, 2>::baseGLType();

// template <>
// int TypeGL<uint64_t, 2>::numGLSLBytes();

template <>
std::string TypeGL<uint64_t, 3>::glslType();

template <>
int TypeGL<uint64_t, 3>::baseGLType();

// template <>
// int TypeGL<uint64_t, 3>::numGLSLBytes();

template <>
std::string TypeGL<uint64_t, 4>::glslType();

template <>
int TypeGL<uint64_t, 4>::baseGLType();

// template <>
// int TypeGL<uint64_t, 4>::numGLSLBytes();

// INT64

template <>
std::string TypeGL<int64_t, 1>::glslType();

template <>
int TypeGL<int64_t, 1>::baseGLType();

// template <>
// int TypeGL<int64_t, 1>::numGLSLBytes();

template <>
std::string TypeGL<int64_t, 2>::glslType();

template <>
int TypeGL<int64_t, 2>::baseGLType();

// template <>
// int TypeGL<int64_t, 2>::numGLSLBytes();

template <>
std::string TypeGL<int64_t, 3>::glslType();

template <>
int TypeGL<int64_t, 3>::baseGLType();

// template <>
// int TypeGL<int64_t, 3>::numGLSLBytes();

template <>
std::string TypeGL<int64_t, 4>::glslType();

template <>
int TypeGL<int64_t, 4>::baseGLType();

// template <>
// int TypeGL<int64_t, 4>::numGLSLBytes();

}  // namespace MapD_renderer

#endif  // TYPE_GL_H_
